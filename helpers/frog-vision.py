#!/usr/bin/env python3
"""Frog-Vision - a single-file neon Frogger-style arcade game for FrameVision.

Run with:  python Frog-vision.py
Optional:  python Frog-vision.py --self-test
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pygame

# -----------------------------------------------------------------------------
# Core layout / palette
# -----------------------------------------------------------------------------
DESIGN_W, DESIGN_H = 1280, 720
FPS = 60
ARENA_X, ARENA_Y = 128, 92
ARENA_W, LANE_H = 1024, 46
COLS, ROWS = 16, 13
CELL_W = ARENA_W // COLS
ARENA_H = ROWS * LANE_H
FINISH_ROW = 0
RIVER_ROWS = (1, 2, 3, 4, 5)
MEDIAN_ROW = 6
ROAD_ROWS = (7, 8, 9, 10, 11)
START_ROW = 12
MAX_LEVEL = 10

WHITE = (235, 248, 255)
BLACK = (4, 7, 14)
CYAN = (32, 232, 255)
BLUE = (40, 108, 255)
TEAL = (0, 210, 165)
LIME = (102, 255, 120)
YELLOW = (255, 224, 70)
ORANGE = (255, 150, 48)
RED = (255, 70, 76)
PURPLE = (142, 88, 255)
ROAD = (13, 20, 34)
WATER = (5, 35, 66)
GRASS = (9, 48, 43)
PANEL = (8, 14, 27)

DIFFICULTIES = {
    "Easy": {
        "vehicle": 0.82, "platform": 0.88, "density": 0.88, "gap": 0.90,
        "timer": 54.0, "lives": 5, "score": 0.85, "bonus": 1.35,
        "enemy": 0.75,
    },
    "Normal": {
        "vehicle": 1.00, "platform": 1.00, "density": 1.00, "gap": 1.00,
        "timer": 46.0, "lives": 4, "score": 1.00, "bonus": 1.00,
        "enemy": 1.00,
    },
    "Hard": {
        "vehicle": 1.18, "platform": 1.12, "density": 1.15, "gap": 1.10,
        "timer": 39.0, "lives": 3, "score": 1.30, "bonus": 0.72,
        "enemy": 1.28,
    },
}

VEHICLE_SPECS = {
    "Compact Car": (82, 31, 145, CYAN),
    "Sports Car": (68, 27, 205, ORANGE),
    "Truck": (142, 35, 105, YELLOW),
    "Bus": (126, 36, 132, TEAL),
    "Motorcycle": (48, 22, 250, LIME),
    "Hazard Transport": (156, 38, 122, RED),
}

PLATFORM_SPECS = {
    "Log": (150, 29, 92),
    "Tree Trunk": (220, 33, 70),
    "Turtle Group": (150, 29, 104),
    "Mechanical Raft": (132, 31, 132),
    "Broken Log": (108, 28, 112),
    "Cyber-Gator": (190, 31, 96),
}

SPECIALS = {
    3: ("Rush Hour", "Traffic lanes surge after a visible warning pulse."),
    5: ("Storm River", "Rain darkens the water; lightning restores full visibility."),
    6: ("Reversing Current", "Selected river lanes warn, slow, then reverse direction."),
    8: ("Night Crossing", "The arena is darker, but hazards gain stronger neon outlines."),
    10: ("Flood Surge", "River lanes periodically accelerate after a cyan warning wave."),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out(t: float) -> float:
    return 1.0 - (1.0 - clamp(t, 0.0, 1.0)) ** 3


def row_y(row: int) -> int:
    return ARENA_Y + row * LANE_H


def row_center(row: int) -> float:
    return row_y(row) + LANE_H / 2


def safe_int(value, default: int, lo: int, hi: int) -> int:
    try:
        return int(clamp(int(value), lo, hi))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float, lo: float, hi: float) -> float:
    try:
        return float(clamp(float(value), lo, hi))
    except (TypeError, ValueError):
        return default


def draw_glow(surface: pygame.Surface, rect: pygame.Rect, color, radius: int = 10, width: int = 0) -> None:
    glow = pygame.Surface((rect.w + radius * 4, rect.h + radius * 4), pygame.SRCALPHA)
    center = pygame.Rect(radius * 2, radius * 2, rect.w, rect.h)
    for r in range(radius, 0, -3):
        alpha = max(5, int(32 * (r / max(1, radius))))
        pygame.draw.rect(glow, (*color, alpha), center.inflate(r * 2, r * 2), width, border_radius=max(4, r))
    pygame.draw.rect(glow, color, center, width, border_radius=7)
    surface.blit(glow, (rect.x - radius * 2, rect.y - radius * 2))


def fill_crop(image: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
    """Scale and center-crop without distortion."""
    sw, sh = image.get_size()
    tw, th = size
    if sw <= 0 or sh <= 0:
        raise ValueError("invalid image size")
    scale = max(tw / sw, th / sh)
    scaled = pygame.transform.smoothscale(image, (max(1, int(sw * scale)), max(1, int(sh * scale))))
    x = (scaled.get_width() - tw) // 2
    y = (scaled.get_height() - th) // 2
    return scaled.subsurface((x, y, tw, th)).copy()


def framevision_paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    # Normal placement: <root>/helpers/Frog-vision.py. The environment override is used only by self-tests.
    test_root = os.environ.get("FROG_VISION_TEST_ROOT")
    root = Path(test_root).resolve() if test_root else (script.parent.parent if script.parent.name.lower() == "helpers" else script.parent)
    return root, root / "presets" / "startup", root / "presets" / "setsave" / "frogger_save.json"


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
class SaveManager:
    VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.data = self.default_data()
        self.load()

    @staticmethod
    def default_stats() -> dict:
        return {
            "high_score": 0,
            "highest_level": 0,
            "total_crossings": 0,
            "successful_slots": 0,
            "total_deaths": 0,
            "vehicle_collisions": 0,
            "water_deaths": 0,
            "timer_deaths": 0,
            "hazard_deaths": 0,
            "bonuses_collected": 0,
            "levels_completed": 0,
            "games_won": 0,
            "games_lost": 0,
            "total_play_time": 0.0,
            "best_level_time": None,
        }

    @classmethod
    def default_data(cls) -> dict:
        return {
            "version": cls.VERSION,
            "settings": {
                "difficulty": "Normal",
                "volume": 0.55,
                "muted": False,
                "fullscreen": False,
                "background_folder": "",
            },
            "stats": cls.default_stats(),
            "campaign": {"active": False},
        }

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        base = self.default_data()
        settings = raw.get("settings", {}) if isinstance(raw.get("settings"), dict) else {}
        difficulty = settings.get("difficulty", "Normal")
        base["settings"] = {
            "difficulty": difficulty if difficulty in DIFFICULTIES else "Normal",
            "volume": safe_float(settings.get("volume"), 0.55, 0.0, 1.0),
            "muted": bool(settings.get("muted", False)),
            "fullscreen": bool(settings.get("fullscreen", False)),
            "background_folder": str(settings.get("background_folder", "")).strip()[:4096],
        }
        stats = raw.get("stats", {}) if isinstance(raw.get("stats"), dict) else {}
        clean_stats = self.default_stats()
        int_fields = [
            "high_score", "highest_level", "total_crossings", "successful_slots",
            "total_deaths", "vehicle_collisions", "water_deaths", "timer_deaths",
            "hazard_deaths", "bonuses_collected", "levels_completed", "games_won", "games_lost",
        ]
        for key in int_fields:
            clean_stats[key] = safe_int(stats.get(key), 0, 0, 2_000_000_000)
        clean_stats["total_play_time"] = safe_float(stats.get("total_play_time"), 0.0, 0.0, 10**9)
        best = stats.get("best_level_time")
        clean_stats["best_level_time"] = None if best is None else safe_float(best, 9999.0, 0.0, 9999.0)
        base["stats"] = clean_stats
        campaign = raw.get("campaign", {}) if isinstance(raw.get("campaign"), dict) else {}
        if campaign.get("active"):
            filled = campaign.get("filled_slots", [])
            if not isinstance(filled, list):
                filled = []
            base["campaign"] = {
                "active": True,
                "level": safe_int(campaign.get("level"), 1, 1, MAX_LEVEL),
                "score": safe_int(campaign.get("score"), 0, 0, 2_000_000_000),
                "lives": safe_int(campaign.get("lives"), DIFFICULTIES[base["settings"]["difficulty"]]["lives"], 1, 99),
                "difficulty": campaign.get("difficulty") if campaign.get("difficulty") in DIFFICULTIES else base["settings"]["difficulty"],
                "filled_slots": sorted({safe_int(v, -1, -1, 4) for v in filled if isinstance(v, (int, float)) and 0 <= int(v) < 5}),
                "next_life_score": safe_int(campaign.get("next_life_score"), 20_000, 20_000, 2_000_000_000),
                "crossing_streak": safe_int(campaign.get("crossing_streak"), 0, 0, 999),
                "deaths_this_level": safe_int(campaign.get("deaths_this_level"), 0, 0, 999),
                "campaign_deaths": safe_int(campaign.get("campaign_deaths"), 0, 0, 999999),
            }
        self.data = base

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.data, indent=2, sort_keys=True)
            fd, tmp_name = tempfile.mkstemp(prefix="frogger_", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            return True
        except OSError:
            return False

    def reset_all(self) -> None:
        self.data = self.default_data()
        self.save()

    @property
    def has_continue(self) -> bool:
        c = self.data.get("campaign", {})
        return bool(c.get("active")) and 1 <= safe_int(c.get("level"), 0, 0, MAX_LEVEL) <= MAX_LEVEL


# -----------------------------------------------------------------------------
# Audio and input
# -----------------------------------------------------------------------------
class SoundManager:
    def __init__(self, volume: float, muted: bool):
        self.available = False
        self.volume = clamp(volume, 0.0, 1.0)
        self.muted = bool(muted)
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.available = True
            self._build_sounds()
            self.apply_volume()
        except (pygame.error, OSError, ValueError):
            self.available = False

    def _tone(self, freq: float, duration: float, volume: float = 0.6, sweep: float = 0.0, noise: float = 0.0) -> pygame.mixer.Sound:
        sample_rate = 22050
        count = max(1, int(sample_rate * duration))
        data = array.array("h")
        phase = 0.0
        rng = random.Random(int(freq * 100 + duration * 1000))
        for i in range(count):
            f = max(20.0, freq + sweep * (i / count))
            phase += 2 * math.pi * f / sample_rate
            env = min(1.0, i / max(1, count * 0.08)) * max(0.0, 1.0 - i / count) ** 1.5
            sample = math.sin(phase) * (1.0 - noise) + rng.uniform(-1, 1) * noise
            data.append(int(clamp(sample * env * volume, -1, 1) * 32767))
        return pygame.mixer.Sound(buffer=data.tobytes())

    def _build_sounds(self) -> None:
        specs = {
            "menu": (540, .045, .28, 90, 0.0), "confirm": (660, .10, .35, 250, 0.0),
            "hop": (380, .08, .34, 220, 0.0), "pass": (150, .12, .22, -35, .12),
            "collision": (95, .24, .52, -55, .42), "splash": (130, .28, .46, -90, .62),
            "land": (300, .07, .25, -80, .08), "finish": (680, .32, .45, 550, 0.0),
            "bonus": (780, .18, .36, 520, 0.0), "warning": (250, .18, .30, 0, .08),
            "tick": (860, .045, .22, 0, 0.0), "level": (420, .55, .42, 720, 0.0),
            "life": (720, .42, .40, 900, 0.0), "gameover": (270, .70, .45, -210, .06),
            "victory": (520, 1.0, .45, 1100, 0.0),
        }
        for name, args in specs.items():
            self.sounds[name] = self._tone(*args)

    def apply_volume(self) -> None:
        if not self.available:
            return
        effective = 0.0 if self.muted else self.volume
        for sound in self.sounds.values():
            sound.set_volume(effective)

    def play(self, name: str) -> None:
        if self.available and not self.muted:
            sound = self.sounds.get(name)
            if sound:
                sound.play()

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self.apply_volume()

    def change_volume(self, delta: float) -> None:
        self.volume = clamp(self.volume + delta, 0.0, 1.0)
        if self.volume > 0 and self.muted:
            self.muted = False
        self.apply_volume()


class InputManager:
    MOVE_KEYS = {
        pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
        pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
        pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
        pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
    }

    def __init__(self):
        self.held: dict[int, float] = {}
        self.repeat_clock: dict[int, float] = {}
        self.order: list[int] = []
        self.delay = 0.26
        self.interval = 0.13
        self.pending: list[tuple[int, int]] = []

    def clear(self) -> None:
        self.held.clear()
        self.repeat_clock.clear()
        self.order.clear()
        self.pending.clear()

    def key_down(self, key: int) -> None:
        if key in self.MOVE_KEYS and key not in self.held:
            self.held[key] = 0.0
            self.repeat_clock[key] = 0.0
            self.order.append(key)
            self.pending.append(self.MOVE_KEYS[key])

    def key_up(self, key: int) -> None:
        self.held.pop(key, None)
        self.repeat_clock.pop(key, None)
        if key in self.order:
            self.order.remove(key)

    def update(self, dt: float) -> None:
        for key in list(self.order):
            if key not in self.held:
                continue
            self.held[key] += dt
            if self.held[key] >= self.delay:
                self.repeat_clock[key] += dt
                while self.repeat_clock[key] >= self.interval:
                    self.repeat_clock[key] -= self.interval
                    # Only the most recently pressed held direction repeats.
                    if self.order and self.order[-1] == key:
                        self.pending.append(self.MOVE_KEYS[key])

    def pop_move(self) -> Optional[tuple[int, int]]:
        return self.pending.pop(0) if self.pending else None


# -----------------------------------------------------------------------------
# Lightweight effects / UI
# -----------------------------------------------------------------------------
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: float
    gravity: float = 0.0

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        return self.life > 0

    def draw(self, surface: pygame.Surface) -> None:
        alpha = int(255 * clamp(self.life / self.max_life, 0, 1))
        pygame.draw.circle(surface, (*self.color, alpha), (int(self.x), int(self.y)), max(1, int(self.size)))


@dataclass
class FloatingText:
    text: str
    x: float
    y: float
    color: tuple[int, int, int]
    life: float = 1.0
    max_life: float = 1.0

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.y -= 30 * dt
        return self.life > 0

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        alpha = int(255 * clamp(self.life / self.max_life, 0, 1))
        img = font.render(self.text, True, self.color)
        img.set_alpha(alpha)
        surface.blit(img, img.get_rect(center=(int(self.x), int(self.y))))


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str
    enabled: bool = True
    tooltip: str = ""

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, mouse: tuple[int, int], selected: bool = False) -> None:
        hover = self.enabled and self.rect.collidepoint(mouse)
        color = CYAN if (hover or selected) else (64, 110, 140)
        fill = (12, 29, 48, 230) if self.enabled else (18, 22, 30, 180)
        pygame.draw.rect(surface, fill, self.rect, border_radius=10)
        draw_glow(surface, self.rect, color if self.enabled else (55, 62, 70), 8, 2)
        txt_color = WHITE if self.enabled else (95, 105, 118)
        txt = font.render(self.label, True, txt_color)
        surface.blit(txt, txt.get_rect(center=self.rect.center))


# -----------------------------------------------------------------------------
# Moving game objects
# -----------------------------------------------------------------------------
class MovingObject:
    _next_id = 1

    def __init__(self, lane_row: int, x: float, width: float, height: float, speed: float, direction: int):
        self.id = MovingObject._next_id
        MovingObject._next_id += 1
        self.row = lane_row
        self.x = float(x)
        self.width = float(width)
        self.height = float(height)
        self.base_speed = float(speed)
        self.direction = 1 if direction >= 0 else -1
        self.prev_x = self.x
        self.dx = 0.0
        self.just_wrapped = False

    @property
    def y(self) -> float:
        return row_center(self.row) - self.height / 2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), round(self.y), round(self.width), round(self.height))

    @property
    def swept_rect(self) -> pygame.Rect:
        old = pygame.Rect(round(self.prev_x), round(self.y), round(self.width), round(self.height))
        return old.union(self.rect)

    def update_motion(self, dt: float, speed_mult: float = 1.0) -> None:
        self.prev_x = self.x
        self.dx = self.direction * self.base_speed * speed_mult * dt
        self.x += self.dx
        self.just_wrapped = False

    def wrap(self, margin: float, lane_span: float, attached: bool = False) -> None:
        left = ARENA_X - margin
        right = ARENA_X + ARENA_W + margin
        if attached:
            return
        if self.direction > 0 and self.x > right:
            self.x -= lane_span
            self.prev_x = self.x
            self.dx = 0
            self.just_wrapped = True
        elif self.direction < 0 and self.x + self.width < left:
            self.x += lane_span
            self.prev_x = self.x
            self.dx = 0
            self.just_wrapped = True


class Vehicle(MovingObject):
    def __init__(self, lane_row: int, x: float, kind: str, speed: float, direction: int):
        w, h, _, color = VEHICLE_SPECS[kind]
        super().__init__(lane_row, x, w, h, speed, direction)
        self.kind = kind
        self.color = color
        self.pass_cooldown = 0.0

    @property
    def collision_rect(self) -> pygame.Rect:
        r = self.rect
        if self.kind == "Motorcycle":
            return r.inflate(-6, -4)
        return r.inflate(-8, -5)

    @property
    def danger_trail(self) -> Optional[pygame.Rect]:
        if self.kind != "Hazard Transport":
            return None
        r = self.rect
        length = 62
        return pygame.Rect(r.left - length if self.direction > 0 else r.right, r.y + 7, length, r.h - 14)

    def draw(self, surface: pygame.Surface, night: bool = False) -> None:
        r = self.rect
        glow_color = tuple(min(255, c + (30 if night else 0)) for c in self.color)
        draw_glow(surface, r, glow_color, 9 if night else 6, 2)
        if self.kind == "Motorcycle":
            pygame.draw.circle(surface, (18, 24, 31), (r.left + 10, r.bottom - 3), 6)
            pygame.draw.circle(surface, (18, 24, 31), (r.right - 10, r.bottom - 3), 6)
            pygame.draw.line(surface, self.color, (r.left + 8, r.centery), (r.right - 7, r.centery), 5)
            pygame.draw.circle(surface, WHITE, (r.right - 3 if self.direction > 0 else r.left + 3, r.centery), 3)
            return
        pygame.draw.rect(surface, (*self.color, 230), r, border_radius=8)
        pygame.draw.rect(surface, (16, 28, 42), r.inflate(-18, -10), border_radius=5)
        if self.kind in ("Truck", "Bus", "Hazard Transport"):
            for x in range(r.left + 18, r.right - 12, 25):
                pygame.draw.rect(surface, (75, 165, 190), (x, r.y + 6, 15, 8), border_radius=2)
        else:
            pygame.draw.rect(surface, (115, 210, 240), (r.centerx - 16, r.y + 5, 32, 9), border_radius=3)
        for wx in (r.left + 17, r.right - 17):
            pygame.draw.circle(surface, (8, 10, 14), (wx, r.bottom), 6)
        head_x = r.right - 5 if self.direction > 0 else r.left + 5
        pygame.draw.circle(surface, WHITE, (head_x, r.centery), 3)
        trail = self.danger_trail
        if trail:
            pulse = 80 + int(55 * (1 + math.sin(pygame.time.get_ticks() * .018)))
            pygame.draw.rect(surface, (*RED, pulse), trail, border_radius=8)
            for x in range(trail.left, trail.right, 14):
                pygame.draw.line(surface, (*YELLOW, pulse), (x, trail.top), (x + 6, trail.bottom), 2)


class Platform(MovingObject):
    def __init__(self, lane_row: int, x: float, kind: str, speed: float, direction: int, phase: float = 0.0):
        w, h, _ = PLATFORM_SPECS[kind]
        super().__init__(lane_row, x, w, h, speed, direction)
        self.kind = kind
        self.phase = phase
        self.age = phase
        self.speed_factor = 1.0
        self.warning = False

    def update(self, dt: float, speed_mult: float, stable: bool = False) -> None:
        self.age += dt
        self.warning = False
        local = 1.0
        if self.kind == "Mechanical Raft":
            cycle = self.age % 6.0
            if 3.8 < cycle < 4.8:
                self.warning = True
            if 4.8 <= cycle < 5.7:
                local = 1.55
        self.speed_factor = local
        self.update_motion(dt, speed_mult * local)

    def turtle_submerge(self, stable: bool = False) -> float:
        if self.kind != "Turtle Group" or stable:
            return 0.0
        cycle = self.age % 7.0
        if cycle < 4.6:
            return 0.0
        if cycle < 5.35:
            return (cycle - 4.6) / .75
        if cycle < 6.1:
            return 1.0
        return 1.0 - (cycle - 6.1) / .9

    def broken_unsafe(self, stable: bool = False) -> bool:
        if self.kind != "Broken Log" or stable:
            return False
        cycle = self.age % 5.5
        self.warning = 3.6 < cycle < 4.25
        return 4.25 <= cycle < 5.0

    def croc_mouth_open(self) -> bool:
        if self.kind != "Cyber-Gator":
            return False
        cycle = self.age % 5.8
        self.warning = 3.7 < cycle < 4.4
        return 4.4 <= cycle < 5.15

    def support_rects(self, stable: bool = False) -> list[pygame.Rect]:
        if self.just_wrapped:
            return []
        r = self.rect
        if self.kind == "Turtle Group" and self.turtle_submerge(stable) >= 0.92:
            return []
        if self.kind == "Broken Log" and self.broken_unsafe(stable):
            return []
        if self.kind == "Cyber-Gator":
            head_w = 48
            if self.direction > 0:
                body = pygame.Rect(r.left + 9, r.top + 3, r.w - head_w - 10, r.h - 6)
            else:
                body = pygame.Rect(r.left + head_w + 1, r.top + 3, r.w - head_w - 10, r.h - 6)
            return [body]
        return [r.inflate(-8, -5)]

    def unsafe_rects(self, stable: bool = False) -> list[pygame.Rect]:
        r = self.rect
        out: list[pygame.Rect] = []
        if self.kind == "Cyber-Gator":
            head_w = 54
            out.append(pygame.Rect(r.right - head_w, r.y, head_w, r.h) if self.direction > 0 else pygame.Rect(r.x, r.y, head_w, r.h))
        if self.kind == "Broken Log" and self.broken_unsafe(stable):
            out.append(r)
        return out

    def draw(self, surface: pygame.Surface, stable: bool = False, night: bool = False) -> None:
        r = self.rect
        if self.kind == "Log":
            draw_glow(surface, r, ORANGE, 5 if not night else 8, 1)
            pygame.draw.rect(surface, (108, 67, 32), r, border_radius=13)
            for x in range(r.left + 15, r.right, 28):
                pygame.draw.line(surface, (170, 110, 54), (x, r.top + 4), (x - 7, r.bottom - 4), 2)
        elif self.kind == "Tree Trunk":
            draw_glow(surface, r, YELLOW, 5 if not night else 8, 1)
            pygame.draw.rect(surface, (92, 61, 33), r, border_radius=14)
            for x in range(r.left + 20, r.right - 10, 35):
                pygame.draw.circle(surface, (153, 101, 51), (x, r.centery), 4, 2)
            pygame.draw.line(surface, (130, 88, 47), (r.left + 60, r.top), (r.left + 75, r.top - 8), 4)
        elif self.kind == "Turtle Group":
            sub = self.turtle_submerge(stable)
            alpha = int(255 * (1 - .65 * sub))
            yoff = int(sub * 15)
            for i in range(3):
                cx = r.left + 25 + i * 50
                pygame.draw.ellipse(surface, (*TEAL, alpha), (cx - 21, r.y + yoff, 42, 23))
                pygame.draw.circle(surface, (*LIME, alpha), (cx + (21 if self.direction > 0 else -21), r.y + 11 + yoff), 6)
            if 0.15 < sub < .92:
                pygame.draw.arc(surface, YELLOW, r.inflate(6, 6), 0, math.tau, 3)
        elif self.kind == "Mechanical Raft":
            col = YELLOW if self.warning else CYAN
            draw_glow(surface, r, col, 8, 2)
            pygame.draw.rect(surface, (26, 66, 82), r, border_radius=7)
            for x in range(r.left + 13, r.right - 10, 24):
                pygame.draw.rect(surface, (79, 160, 178), (x, r.y + 6, 15, r.h - 12), border_radius=3)
            pygame.draw.circle(surface, col, r.center, 5)
        elif self.kind == "Broken Log":
            unsafe = self.broken_unsafe(stable)
            col = RED if unsafe else (YELLOW if self.warning else ORANGE)
            draw_glow(surface, r, col, 7, 2)
            pygame.draw.polygon(surface, (105, 65, 34), [(r.left, r.centery), (r.left + 16, r.top), (r.right - 12, r.top + 4), (r.right, r.centery), (r.right - 18, r.bottom), (r.left + 12, r.bottom - 3)])
            pygame.draw.line(surface, col, (r.centerx - 8, r.top + 2), (r.centerx + 8, r.bottom - 2), 3)
        else:  # Cyber-Gator
            mouth = self.croc_mouth_open()
            col = RED if (mouth or self.warning) else LIME
            draw_glow(surface, r, col, 7, 2)
            pygame.draw.rect(surface, (25, 92, 62), r, border_radius=12)
            head = pygame.Rect(r.right - 54, r.y, 54, r.h) if self.direction > 0 else pygame.Rect(r.x, r.y, 54, r.h)
            pygame.draw.rect(surface, (37, 132, 78), head, border_radius=8)
            eye_x = head.right - 13 if self.direction > 0 else head.left + 13
            pygame.draw.circle(surface, RED, (eye_x, head.top + 8), 3)
            if mouth:
                pygame.draw.line(surface, BLACK, (head.left + 6, head.centery), (head.right - 6, head.centery), 5)
                for tx in range(head.left + 10, head.right - 5, 10):
                    pygame.draw.polygon(surface, WHITE, [(tx, head.centery), (tx + 4, head.centery + 6), (tx + 7, head.centery)])
            for x in range(r.left + 25, r.right - 55, 28):
                pygame.draw.polygon(surface, (66, 174, 105), [(x, r.top), (x + 8, r.top - 7), (x + 16, r.top)])

# -----------------------------------------------------------------------------
# Player, bonuses, hazards, lanes
# -----------------------------------------------------------------------------
class Player:
    def __init__(self):
        self.spawn_x = ARENA_X + ARENA_W / 2 - 18
        self.x = self.spawn_x
        self.row = START_ROW
        self.start_x = self.x
        self.start_row = self.row
        self.target_x = self.x
        self.target_row = self.row
        self.hopping = False
        self.hop_t = 0.0
        self.hop_duration = 0.115
        self.attached: Optional[Platform] = None
        self.invulnerable = 0.0
        self.facing = (0, -1)

    @property
    def y(self) -> float:
        return row_center(self.row) - 17

    @property
    def visual_rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), round(self.y), 36, 34)

    @property
    def body_rect(self) -> pygame.Rect:
        return self.visual_rect.inflate(-12, -9)

    def reset(self) -> None:
        self.x = self.spawn_x
        self.row = START_ROW
        self.start_x = self.x
        self.start_row = self.row
        self.target_x = self.x
        self.target_row = self.row
        self.hopping = False
        self.hop_t = 0.0
        self.attached = None
        self.invulnerable = 1.15
        self.facing = (0, -1)

    def try_hop(self, dx: int, dy: int) -> bool:
        if self.hopping:
            return False
        if dx and dy:
            return False
        new_row = int(clamp(self.row + dy, FINISH_ROW, START_ROW))
        new_x = clamp(self.x + dx * CELL_W, ARENA_X + 4, ARENA_X + ARENA_W - 40)
        if new_row == self.row and abs(new_x - self.x) < 1:
            return False
        self.start_x, self.start_row = self.x, self.row
        self.target_x, self.target_row = new_x, new_row
        self.hop_t = 0.0
        self.hopping = True
        self.attached = None
        self.facing = (dx, dy)
        return True

    def update(self, dt: float) -> bool:
        self.invulnerable = max(0.0, self.invulnerable - dt)
        landed = False
        if self.hopping:
            self.hop_t += dt / self.hop_duration
            t = ease_out(self.hop_t)
            self.x = lerp(self.start_x, self.target_x, t)
            # Row is discrete for collision / lane ownership, while draw adds an arc.
            if self.hop_t >= 0.5:
                self.row = self.target_row
            if self.hop_t >= 1.0:
                self.x = self.target_x
                self.row = self.target_row
                self.hopping = False
                landed = True
        return landed

    def carry(self, dx: float) -> None:
        if not self.hopping:
            self.x += dx
            self.target_x = self.x
            self.start_x = self.x

    def draw(self, surface: pygame.Surface, shield: bool = False) -> None:
        r = self.visual_rect.copy()
        if self.hopping:
            arc = math.sin(clamp(self.hop_t, 0, 1) * math.pi) * 13
            r.y -= int(arc)
        blink = self.invulnerable > 0 and int(self.invulnerable * 12) % 2 == 0
        if blink:
            return
        draw_glow(surface, r, CYAN if shield else LIME, 9, 2)
        pygame.draw.ellipse(surface, (36, 205, 92), r)
        pygame.draw.ellipse(surface, (71, 255, 135), (r.x + 7, r.y + 5, r.w - 14, r.h - 10))
        # Legs and eyes make the silhouette readable without an image asset.
        pygame.draw.ellipse(surface, (25, 160, 75), (r.x - 6, r.y + 19, 15, 11))
        pygame.draw.ellipse(surface, (25, 160, 75), (r.right - 9, r.y + 19, 15, 11))
        pygame.draw.circle(surface, WHITE, (r.x + 11, r.y + 7), 5)
        pygame.draw.circle(surface, WHITE, (r.right - 11, r.y + 7), 5)
        pygame.draw.circle(surface, BLACK, (r.x + 12, r.y + 7), 2)
        pygame.draw.circle(surface, BLACK, (r.right - 10, r.y + 7), 2)
        if shield:
            pulse = 2 + int(2 * (1 + math.sin(pygame.time.get_ticks() * .012)))
            pygame.draw.ellipse(surface, (*CYAN, 150), r.inflate(10 + pulse, 10 + pulse), 2)


@dataclass
class Bonus:
    kind: str
    row: int
    x: float
    life: float = 9.0
    slot_index: Optional[int] = None
    collected: bool = False

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x - 14), int(row_center(self.row) - 14), 28, 28)

    def update(self, dt: float) -> bool:
        self.life -= dt
        return self.life > 0 and not self.collected

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        pulse = 1.0 + .16 * math.sin(pygame.time.get_ticks() * .012)
        r = self.rect.inflate(int(8 * pulse), int(8 * pulse))
        colors = {
            "Time Extension": CYAN, "Extra Life": LIME, "Score Multiplier": YELLOW,
            "Temporary Shield": BLUE, "Slow Traffic": ORANGE, "Stable Platforms": TEAL,
            "Bonus Fly": PURPLE,
        }
        symbols = {
            "Time Extension": "+T", "Extra Life": "+1", "Score Multiplier": "2X",
            "Temporary Shield": "S", "Slow Traffic": "SL", "Stable Platforms": "ST",
            "Bonus Fly": "FLY",
        }
        color = colors[self.kind]
        draw_glow(surface, r, color, 8, 2)
        pygame.draw.circle(surface, (12, 24, 36), r.center, max(11, r.w // 2 - 3))
        txt = font.render(symbols[self.kind], True, color)
        surface.blit(txt, txt.get_rect(center=r.center))
        # Expiry ring.
        ratio = clamp(self.life / 9.0, 0, 1)
        pygame.draw.arc(surface, WHITE, r.inflate(4, 4), -math.pi / 2, -math.pi / 2 + math.tau * ratio, 2)


class Hazard:
    def __init__(self, kind: str, row: int, x: float, direction: int = 1):
        self.kind = kind
        self.row = row
        self.x = x
        self.direction = direction
        self.age = 0.0
        self.life = 999.0
        self.target: Optional[tuple[float, float]] = None
        self.phase = "move"
        self.strike_timer = 0.0

    @property
    def rect(self) -> pygame.Rect:
        if self.kind == "Snake":
            return pygame.Rect(int(self.x), int(row_center(self.row) - 10), 68, 20)
        if self.kind == "Otter Drone":
            return pygame.Rect(int(self.x), int(row_center(self.row) - 13), 54, 26)
        if self.kind == "Aerial Drone" and self.target:
            return pygame.Rect(int(self.target[0] - 25), int(self.target[1] - 25), 50, 50)
        return pygame.Rect(int(self.x), int(row_center(self.row) - 12), 44, 24)

    def update(self, dt: float, player: Player, speed_mult: float = 1.0) -> None:
        self.age += dt
        if self.kind == "Snake":
            self.x += self.direction * 66 * speed_mult * dt
            if self.direction > 0 and self.x > ARENA_X + ARENA_W + 80:
                self.x = ARENA_X - 100
            elif self.direction < 0 and self.x < ARENA_X - 100:
                self.x = ARENA_X + ARENA_W + 80
        elif self.kind == "Otter Drone":
            self.x += self.direction * 118 * speed_mult * dt
            if self.direction > 0 and self.x > ARENA_X + ARENA_W + 60:
                self.x = ARENA_X - 80
            elif self.direction < 0 and self.x < ARENA_X - 80:
                self.x = ARENA_X + ARENA_W + 60
        elif self.kind == "Aerial Drone":
            if self.phase == "cooldown":
                self.strike_timer -= dt
                if self.strike_timer <= 0 and player.row not in (START_ROW, FINISH_ROW):
                    self.target = player.body_rect.center
                    self.phase = "warning"
                    self.strike_timer = 1.45
            elif self.phase == "warning":
                self.strike_timer -= dt
                if self.strike_timer <= 0:
                    self.phase = "strike"
                    self.strike_timer = .22
            elif self.phase == "strike":
                self.strike_timer -= dt
                if self.strike_timer <= 0:
                    self.phase = "cooldown"
                    self.strike_timer = 5.5
                    self.target = None

    def dangerous(self) -> bool:
        return self.kind != "Aerial Drone" or self.phase == "strike"

    def draw(self, surface: pygame.Surface) -> None:
        if self.kind == "Snake":
            r = self.rect
            points = []
            for i in range(7):
                px = r.left + i * 10
                py = r.centery + int(math.sin(self.age * 7 + i) * 6)
                points.append((px, py))
            pygame.draw.lines(surface, LIME, False, points, 7)
            pygame.draw.circle(surface, RED, points[-1], 3)
        elif self.kind == "Otter Drone":
            r = self.rect
            draw_glow(surface, r, ORANGE, 5, 1)
            pygame.draw.ellipse(surface, (113, 72, 40), r)
            pygame.draw.circle(surface, (155, 104, 55), (r.right - 8 if self.direction > 0 else r.left + 8, r.centery), 9)
            pygame.draw.circle(surface, CYAN, (r.centerx, r.centery), 4)
        elif self.kind == "Aerial Drone" and self.target:
            cx, cy = self.target
            color = RED if self.phase == "strike" else YELLOW
            radius = 34 + int(5 * math.sin(self.age * 12))
            pygame.draw.circle(surface, (*color, 100), (int(cx), int(cy)), radius, 3)
            pygame.draw.line(surface, (*color, 130), (int(cx - radius), int(cy)), (int(cx + radius), int(cy)), 2)
            pygame.draw.line(surface, (*color, 130), (int(cx), int(cy - radius)), (int(cx), int(cy + radius)), 2)
            if self.phase == "strike":
                pygame.draw.circle(surface, (*WHITE, 220), (int(cx), int(cy)), 22)


class Lane:
    def __init__(self, row: int, direction: int, speed: float, spacing: float):
        self.row = row
        self.direction = direction
        self.speed = speed
        self.spacing = spacing


class RoadLane(Lane):
    def __init__(self, row: int, direction: int, speed: float, spacing: float, kinds: list[str], level: int, rng: random.Random):
        super().__init__(row, direction, speed, spacing)
        self.vehicles: list[Vehicle] = []
        span = ARENA_W + spacing * 2
        count = max(2, int(span / spacing) + 1)
        phase = rng.uniform(0, spacing)
        for i in range(count):
            kind = rng.choice(kinds)
            _, _, base, _ = VEHICLE_SPECS[kind]
            local_speed = speed * (base / 145.0) * rng.uniform(.94, 1.07)
            x = ARENA_X - spacing + phase + i * spacing
            self.vehicles.append(Vehicle(row, x, kind, local_speed, direction))
        self.span = max(span, count * spacing)

    def update(self, dt: float, speed_mult: float) -> None:
        for v in self.vehicles:
            v.update_motion(dt, speed_mult)
            v.wrap(220, self.span)

    def clear_spawn_area(self) -> None:
        # The start row itself is safe; push any unusually near road object away from the center lane entry.
        entry = pygame.Rect(int(ARENA_X + ARENA_W / 2 - 70), row_y(self.row), 140, LANE_H)
        for v in self.vehicles:
            if v.rect.colliderect(entry) and self.row == ROAD_ROWS[-1]:
                v.x += -210 if v.direction > 0 else 210
                v.prev_x = v.x

    def draw(self, surface: pygame.Surface, night: bool = False) -> None:
        for v in self.vehicles:
            v.draw(surface, night)


class RiverLane(Lane):
    def __init__(self, row: int, direction: int, speed: float, spacing: float, kinds: list[str], level: int, rng: random.Random):
        super().__init__(row, direction, speed, spacing)
        self.platforms: list[Platform] = []
        span = ARENA_W + spacing * 2
        count = max(2, int(span / spacing) + 1)
        phase = rng.uniform(0, spacing)
        for i in range(count):
            kind = rng.choice(kinds)
            _, _, base = PLATFORM_SPECS[kind]
            local_speed = speed * (base / 92.0) * rng.uniform(.94, 1.06)
            x = ARENA_X - spacing + phase + i * spacing
            self.platforms.append(Platform(row, x, kind, local_speed, direction, rng.uniform(0, 6)))
        self.span = max(span, count * spacing)
        self.reversal_warning = 0.0

    def update(self, dt: float, speed_mult: float, stable: bool, attached: Optional[Platform]) -> None:
        for p in self.platforms:
            p.update(dt, speed_mult, stable)
            p.wrap(260, self.span, attached is p)

    def reverse(self) -> None:
        self.direction *= -1
        for p in self.platforms:
            p.direction *= -1

    def draw(self, surface: pygame.Surface, stable: bool = False, night: bool = False) -> None:
        for p in self.platforms:
            p.draw(surface, stable, night)


class LevelWorld:
    def __init__(self, level: int, difficulty: str, seed: Optional[int] = None):
        self.level = level
        self.difficulty = difficulty
        self.cfg = DIFFICULTIES[difficulty]
        self.rng = random.Random(seed if seed is not None else time.time_ns())
        self.road_lanes: list[RoadLane] = []
        self.river_lanes: list[RiverLane] = []
        self.hazards: list[Hazard] = []
        self.bonus: Optional[Bonus] = None
        self.bonus_timer = self.rng.uniform(7.0, 11.0) / self.cfg["bonus"]
        self.special_name, self.special_desc = SPECIALS.get(level, ("Standard Crossing", "No special modifier. Read the traffic and current."))
        self.special_clock = 0.0
        self.special_warning = 0.0
        self.special_active = False
        self.reversal_next = 6.5
        self.reversal_lanes: list[int] = []
        self.finish_blocker_phase = self.rng.random() * math.tau
        self._build()

    def _vehicle_pool(self, row_index: int) -> list[str]:
        pool = ["Compact Car", "Sports Car", "Truck", "Bus"]
        if self.level >= 3 and row_index >= 2:
            pool.append("Motorcycle")
        if self.level >= 6:
            pool.append("Motorcycle")
        if self.level >= 8 and row_index in (0, 3, 4):
            pool.append("Hazard Transport")
        return pool

    def _platform_pool(self, row_index: int) -> list[str]:
        pools = [
            ["Log", "Tree Trunk"],
            ["Turtle Group", "Log"],
            ["Tree Trunk", "Mechanical Raft"],
            ["Log", "Broken Log"],
            ["Turtle Group", "Cyber-Gator"],
        ]
        pool = list(pools[row_index])
        if self.level < 3:
            pool = [p for p in pool if p not in ("Mechanical Raft", "Broken Log", "Cyber-Gator")]
            if not pool:
                pool = ["Log"]
        elif self.level < 5:
            pool = [p for p in pool if p != "Cyber-Gator"]
        if self.level >= 6 and "Turtle Group" in pool:
            pool.extend(["Turtle Group"] * (1 + self.level // 5))
        if self.level >= 8 and "Cyber-Gator" in pool:
            pool.extend(["Cyber-Gator"] * 2)
        return pool

    def _build(self) -> None:
        level_scale = 1.0 + (self.level - 1) * .055
        for i, row in enumerate(ROAD_ROWS):
            direction = 1 if i % 2 == 0 else -1
            base_speed = (128 + i * 9) * level_scale * self.cfg["vehicle"]
            spacing = (292 - self.level * 6 - i * 3) / self.cfg["density"]
            spacing = max(205, spacing)
            self.road_lanes.append(RoadLane(row, direction, base_speed, spacing, self._vehicle_pool(i), self.level, self.rng))
        for i, row in enumerate(RIVER_ROWS):
            direction = -1 if i % 2 == 0 else 1
            base_speed = (78 + i * 7) * (1 + (self.level - 1) * .045) * self.cfg["platform"]
            # River spacing is constrained against platform length so every lane remains crossable.
            spacing = (235 + i * 7 + self.level * 2) * self.cfg["gap"]
            spacing = clamp(spacing, 205, 292)
            self.river_lanes.append(RiverLane(row, direction, base_speed, spacing, self._platform_pool(i), self.level, self.rng))
        if self.level >= 3:
            self.hazards.append(Hazard("Snake", MEDIAN_ROW, ARENA_X - 90, 1))
        if self.level >= 5:
            self.hazards.append(Hazard("Otter Drone", self.rng.choice(RIVER_ROWS), ARENA_X + ARENA_W + 40, -1))
        if self.level >= 7:
            drone = Hazard("Aerial Drone", -1, 0)
            drone.phase = "cooldown"
            drone.strike_timer = 4.0
            self.hazards.append(drone)

    def update_special(self, dt: float, player: Player) -> tuple[float, float]:
        self.special_clock += dt
        road_mult = river_mult = 1.0
        self.special_warning = 0.0
        self.special_active = False
        if self.special_name == "Rush Hour":
            cycle = self.special_clock % 9.0
            if 5.6 < cycle < 6.6:
                self.special_warning = 1.0 - abs(6.1 - cycle) / .5
            elif 6.6 <= cycle < 8.3:
                road_mult = 1.52
                self.special_active = True
        elif self.special_name == "Reversing Current":
            self.reversal_next -= dt
            if 0 < self.reversal_next < 1.25:
                if not self.reversal_lanes:
                    choices = list(range(len(self.river_lanes)))
                    self.rng.shuffle(choices)
                    self.reversal_lanes = choices[:2]
                self.special_warning = 1.0
                for idx in self.reversal_lanes:
                    self.river_lanes[idx].reversal_warning = self.reversal_next
            elif self.reversal_next <= 0:
                for idx in self.reversal_lanes:
                    self.river_lanes[idx].reverse()
                    self.river_lanes[idx].reversal_warning = 0.0
                self.reversal_lanes = []
                self.reversal_next = 7.0
                self.special_active = True
        elif self.special_name == "Flood Surge":
            cycle = self.special_clock % 8.0
            if 4.6 < cycle < 5.5:
                self.special_warning = 1.0
            elif 5.5 <= cycle < 7.1:
                river_mult = 1.58
                self.special_active = True
        return road_mult, river_mult

    def update(self, dt: float, player: Player, effects: dict[str, float]) -> None:
        road_mult, river_mult = self.update_special(dt, player)
        if effects.get("Slow Traffic", 0) > 0:
            road_mult *= .62
        stable = effects.get("Stable Platforms", 0) > 0
        for lane in self.road_lanes:
            lane.update(dt, road_mult)
        for lane in self.river_lanes:
            lane.update(dt, river_mult, stable, player.attached)
        for hazard in self.hazards:
            hazard.update(dt, player, self.cfg["enemy"])
        self.bonus_timer -= dt
        if self.bonus is not None and not self.bonus.update(dt):
            self.bonus = None
        if self.bonus is None and self.bonus_timer <= 0:
            self.spawn_bonus()
            self.bonus_timer = self.rng.uniform(10.0, 15.0) / self.cfg["bonus"]
        self.finish_blocker_phase += dt * (0.7 + self.level * .05)

    def spawn_bonus(self) -> None:
        kinds = ["Time Extension", "Score Multiplier", "Temporary Shield", "Slow Traffic", "Stable Platforms"]
        if self.rng.random() < .16:
            kinds.append("Extra Life")
        kind = self.rng.choice(kinds)
        # Median and start bank are always reachable and never overlap vehicles or water gaps.
        row = MEDIAN_ROW if self.rng.random() < .82 else START_ROW
        col = self.rng.randint(1, COLS - 2)
        x = ARENA_X + col * CELL_W + CELL_W / 2
        self.bonus = Bonus(kind, row, x)

    def maybe_spawn_finish_fly(self, filled: set[int], slot_centers: list[float]) -> None:
        if self.bonus is not None or self.rng.random() > .18:
            return
        open_slots = [i for i in range(5) if i not in filled]
        if open_slots:
            idx = self.rng.choice(open_slots)
            self.bonus = Bonus("Bonus Fly", FINISH_ROW, slot_centers[idx], 8.0, idx)

    def finish_blockers(self) -> list[pygame.Rect]:
        if self.level < 6:
            return []
        y = row_y(FINISH_ROW) + 8
        width = 50
        span = ARENA_W - width
        x1 = ARENA_X + (math.sin(self.finish_blocker_phase) * .5 + .5) * span
        blockers = [pygame.Rect(int(x1), y, width, LANE_H - 16)]
        if self.level >= 9:
            x2 = ARENA_X + (math.sin(self.finish_blocker_phase + math.pi) * .5 + .5) * span
            blockers.append(pygame.Rect(int(x2), y, width, LANE_H - 16))
        return blockers

    def road_lane_for_row(self, row: int) -> Optional[RoadLane]:
        return next((lane for lane in self.road_lanes if lane.row == row), None)

    def river_lane_for_row(self, row: int) -> Optional[RiverLane]:
        return next((lane for lane in self.river_lanes if lane.row == row), None)

    def clear_spawn_area(self) -> None:
        for lane in self.road_lanes:
            lane.clear_spawn_area()
        self.hazards = [h for h in self.hazards if not (h.kind == "Snake" and abs(h.x - (ARENA_X + ARENA_W / 2)) < 140)] + [
            h for h in []
        ]

    def draw_objects(self, surface: pygame.Surface, effects: dict[str, float]) -> None:
        night = self.special_name == "Night Crossing"
        stable = effects.get("Stable Platforms", 0) > 0
        for lane in self.river_lanes:
            lane.draw(surface, stable, night)
        for lane in self.road_lanes:
            lane.draw(surface, night)
        for hazard in self.hazards:
            hazard.draw(surface)

# -----------------------------------------------------------------------------
# Game controller / state machine
# -----------------------------------------------------------------------------
class Game:
    def __init__(self, headless: bool = False):
        pygame.init()
        self.headless = headless
        self.root, self.startup_dir, save_path = framevision_paths()
        self.save = SaveManager(save_path)
        self.settings = self.save.data["settings"]
        self.windowed_size = (DESIGN_W, DESIGN_H)
        self.fullscreen = bool(self.settings["fullscreen"]) and not headless
        flags = pygame.DOUBLEBUF | pygame.RESIZABLE
        if self.fullscreen:
            flags = pygame.FULLSCREEN | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode((0, 0), flags)
        else:
            size = (640, 360) if headless else self.windowed_size
            self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("Frog-Vision")
        self.canvas = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        self.overlay = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "tiny": pygame.font.SysFont("consolas", 15, bold=True),
            "small": pygame.font.SysFont("consolas", 19, bold=True),
            "body": pygame.font.SysFont("consolas", 24, bold=True),
            "menu": pygame.font.SysFont("consolas", 28, bold=True),
            "title": pygame.font.SysFont("consolas", 62, bold=True),
            "huge": pygame.font.SysFont("consolas", 76, bold=True),
        }
        self.sound = SoundManager(self.settings["volume"], self.settings["muted"])
        self.input = InputManager()
        self.background_path: Optional[Path] = None
        self.background_source = "Generated"
        self.background_notice = ""
        self.background = self.load_background()
        self.background_name = getattr(self, "background_name", "Generated fallback")
        self.state = "title"
        self.return_state = "title"
        self.confirm_action = ""
        self.running = True
        self.selected_index = 0
        self.buttons: list[Button] = []
        self.mouse_design = (-999, -999)
        self.particles: list[Particle] = []
        self.floaters: list[FloatingText] = []
        self.ambient: list[Particle] = [
            Particle(random.uniform(0, DESIGN_W), random.uniform(0, DESIGN_H), random.uniform(-8, 8), random.uniform(-16, -4), random.uniform(2, 6), 6, random.choice([CYAN, BLUE, TEAL]), random.uniform(1, 3))
            for _ in range(70)
        ]
        self.play_session_start = time.monotonic()
        self.last_stats_save = 0.0
        self.screen_shake = 0.0
        self.flash = 0.0
        self.flash_color = WHITE
        self.message = ""
        self.message_timer = 0.0
        self.low_timer_tick = -1
        self.level_summary: dict = {}
        self.death_reason = ""
        self.death_timer = 0.0
        self.level_elapsed = 0.0
        self.level_bonus_awarded = False
        self.game_result_recorded = False
        self.world: Optional[LevelWorld] = None
        self.player = Player()
        self.effects: dict[str, float] = {}
        self.filled_slots: set[int] = set()
        self.slot_centers = [ARENA_X + ARENA_W * (i + .5) / 5 for i in range(5)]
        self.level = 1
        self.score = 0
        self.lives = DIFFICULTIES[self.settings["difficulty"]]["lives"]
        self.difficulty = self.settings["difficulty"]
        self.next_life_score = 20_000
        self.crossing_streak = 0
        self.deaths_this_level = 0
        self.campaign_deaths = 0
        self.timer_max = 46.0
        self.timer = self.timer_max
        self.best_row = START_ROW
        self.build_title_buttons()

    # --------------------------- setup / persistence -------------------------
    @staticmethod
    def image_candidates(folder: Path) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        try:
            return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        except OSError:
            return []

    def custom_background_dir(self) -> Optional[Path]:
        raw = str(self.settings.get("background_folder", "")).strip()
        if not raw:
            return None
        try:
            return Path(os.path.expandvars(raw)).expanduser()
        except (OSError, ValueError):
            return None

    def configured_background_dir(self) -> Path:
        custom = self.custom_background_dir()
        if custom is not None and custom.is_dir():
            return custom
        return self.startup_dir

    def _load_from_background_dir(self, folder: Path, source: str) -> Optional[pygame.Surface]:
        candidates = self.image_candidates(folder)
        previous = self.background_path
        if previous is not None and len(candidates) > 1:
            previous_resolved = previous.resolve()
            alternatives = [p for p in candidates if p.resolve() != previous_resolved]
            if alternatives:
                candidates = alternatives
        random.shuffle(candidates)
        for image_path in candidates:
            try:
                image = pygame.image.load(str(image_path)).convert()
                self.background_path = image_path.resolve()
                self.background_name = image_path.name
                self.background_source = source
                return fill_crop(image, (DESIGN_W, DESIGN_H))
            except (pygame.error, OSError, ValueError):
                continue
        return None

    def load_background(self) -> pygame.Surface:
        custom = self.custom_background_dir()
        if custom is not None and custom.is_dir():
            loaded = self._load_from_background_dir(custom, "Custom")
            if loaded is not None:
                return loaded
        # Keep the original FrameVision startup folder as the reliable default.
        if custom is None or custom.resolve() != self.startup_dir.resolve():
            loaded = self._load_from_background_dir(self.startup_dir, "Default")
            if loaded is not None:
                return loaded
        self.background_path = None
        self.background_name = "Generated fallback"
        self.background_source = "Generated"
        surf = pygame.Surface((DESIGN_W, DESIGN_H))
        for y in range(DESIGN_H):
            t = y / DESIGN_H
            color = (int(5 + 8 * t), int(10 + 19 * t), int(25 + 32 * t))
            pygame.draw.line(surf, color, (0, y), (DESIGN_W, y))
        rng = random.Random(7717)
        for _ in range(95):
            x, y = rng.randrange(DESIGN_W), rng.randrange(DESIGN_H)
            pygame.draw.circle(surf, rng.choice([(15, 90, 112), (18, 54, 102), (12, 110, 85)]), (x, y), rng.randrange(1, 5))
        for x in range(-200, DESIGN_W + 300, 150):
            pygame.draw.polygon(surf, (7, 24, 44), [(x, DESIGN_H), (x + 120, 250), (x + 250, DESIGN_H)])
        return surf

    def choose_background_folder(self) -> None:
        if self.headless:
            self.background_notice = "Folder selection is unavailable in headless mode."
            return
        initial = self.configured_background_dir()
        if not initial.is_dir():
            initial = self.root
        selected = ""
        try:
            import tkinter as tk
            from tkinter import filedialog

            dialog = tk.Tk()
            dialog.withdraw()
            dialog.attributes("-topmost", True)
            dialog.update()
            selected = filedialog.askdirectory(
                parent=dialog,
                title="Select Frog-Vision background image folder",
                initialdir=str(initial),
                mustexist=True,
            )
            dialog.destroy()
        except Exception as exc:
            self.background_notice = f"Could not open folder selector: {exc}"
            return
        if not selected:
            self.background_notice = "Background folder was not changed."
            return
        folder = Path(selected)
        candidates = self.image_candidates(folder)
        if not candidates:
            self.background_notice = "No PNG, JPG, BMP, or WEBP images found; folder unchanged."
            return
        self.settings["background_folder"] = str(folder.resolve())
        self.background = self.load_background()
        self.background_notice = f"Custom folder selected: {len(candidates)} supported image(s)."
        self.write_settings()

    def use_default_background_folder(self) -> None:
        self.settings["background_folder"] = ""
        self.background = self.load_background()
        self.background_notice = "Using FrameVision presets/startup again."
        self.write_settings()

    def write_settings(self) -> None:
        self.settings["difficulty"] = self.difficulty
        self.settings["volume"] = round(self.sound.volume, 3)
        self.settings["muted"] = self.sound.muted
        self.settings["fullscreen"] = self.fullscreen
        self.settings["background_folder"] = str(self.settings.get("background_folder", "")).strip()
        self.save.save()

    def campaign_snapshot(self, active: bool = True) -> dict:
        # A completed level is persisted as the next safe level start, never as a
        # half-transition with five already-filled slots. Confirmation dialogs
        # retain their originating state through return_state.
        effective_state = self.return_state if self.state == "confirm" else self.state
        snap_level = self.level
        snap_filled = sorted(self.filled_slots)
        snap_deaths = self.deaths_this_level
        if effective_state == "level_complete":
            if self.level < MAX_LEVEL:
                snap_level = self.level + 1
                snap_filled = []
                snap_deaths = 0
            else:
                active = False
        if self.lives <= 0:
            active = False
        return {
            "active": active,
            "level": snap_level,
            "score": int(max(0, self.score)),
            "lives": int(max(1, self.lives)),
            "difficulty": self.difficulty,
            "filled_slots": snap_filled,
            "next_life_score": self.next_life_score,
            "crossing_streak": self.crossing_streak,
            "deaths_this_level": snap_deaths,
            "campaign_deaths": self.campaign_deaths,
        }

    def save_campaign(self, active: bool = True) -> None:
        stats = self.save.data["stats"]
        stats["high_score"] = max(stats["high_score"], int(self.score))
        stats["highest_level"] = max(stats["highest_level"], int(self.level))
        self.save.data["campaign"] = self.campaign_snapshot(active)
        self.write_settings()

    def flush_play_time(self) -> None:
        now = time.monotonic()
        delta = max(0.0, now - self.play_session_start)
        self.play_session_start = now
        self.save.data["stats"]["total_play_time"] += delta

    def new_game(self) -> None:
        self.difficulty = self.settings["difficulty"]
        self.level = 1
        self.score = 0
        self.lives = int(DIFFICULTIES[self.difficulty]["lives"])
        self.next_life_score = 20_000
        self.crossing_streak = 0
        self.deaths_this_level = 0
        self.campaign_deaths = 0
        self.filled_slots = set()
        self.effects.clear()
        self.game_result_recorded = False
        self.start_level(intro=True)
        self.save_campaign(True)

    def continue_game(self) -> None:
        c = self.save.data.get("campaign", {})
        if not self.save.has_continue:
            self.new_game()
            return
        self.level = safe_int(c.get("level"), 1, 1, MAX_LEVEL)
        self.score = safe_int(c.get("score"), 0, 0, 2_000_000_000)
        self.difficulty = c.get("difficulty") if c.get("difficulty") in DIFFICULTIES else "Normal"
        self.lives = safe_int(c.get("lives"), DIFFICULTIES[self.difficulty]["lives"], 1, 99)
        self.filled_slots = set(c.get("filled_slots", []))
        self.next_life_score = safe_int(c.get("next_life_score"), 20_000, 20_000, 2_000_000_000)
        self.crossing_streak = safe_int(c.get("crossing_streak"), 0, 0, 999)
        self.deaths_this_level = safe_int(c.get("deaths_this_level"), 0, 0, 999)
        self.campaign_deaths = safe_int(c.get("campaign_deaths"), 0, 0, 999999)
        self.effects.clear()
        self.game_result_recorded = False
        self.start_level(intro=True, preserve_slots=True)

    def start_level(self, intro: bool = True, preserve_slots: bool = False) -> None:
        if not preserve_slots:
            self.filled_slots.clear()
        # Every level gets a fresh random background. When multiple images are
        # available, load_background deliberately avoids the previous image.
        self.background = self.load_background()
        self.world = LevelWorld(self.level, self.difficulty)
        self.player.reset()
        self.player.invulnerable = .7
        self.input.clear()
        self.effects.clear()
        self.timer_max = max(25.0, DIFFICULTIES[self.difficulty]["timer"] - (self.level - 1) * 1.45)
        self.timer = self.timer_max
        self.best_row = START_ROW
        self.level_elapsed = 0.0
        self.level_bonus_awarded = False
        self.low_timer_tick = -1
        self.death_timer = 0.0
        self.death_reason = ""
        self.particles.clear()
        self.floaters.clear()
        if self.world:
            self.world.clear_spawn_area()
            self.world.maybe_spawn_finish_fly(self.filled_slots, self.slot_centers)
        self.state = "level_intro" if intro else "playing"
        self.selected_index = 0

    # ------------------------------- UI models -------------------------------
    def build_title_buttons(self) -> None:
        labels: list[tuple[str, str, bool, str]] = [
            ("NEW GAME", "new", True, "Start a fresh ten-level campaign."),
        ]
        if self.save.has_continue:
            labels.append(("CONTINUE GAME", "continue", True, "Resume at a safe level-start state."))
        background_mode = "CUSTOM" if self.settings.get("background_folder") else "DEFAULT"
        labels += [
            ("INSTRUCTIONS", "instructions", True, "Controls, hazards, bonuses, and scoring."),
            (f"DIFFICULTY: {self.settings['difficulty'].upper()}", "difficulty", True, "Changes speed, density, timer, lives, scoring, and bonuses."),
            ("STATISTICS", "statistics", True, "Persistent lifetime statistics."),
            ("HIGH SCORE", "highscores", True, "View the best score and campaign records."),
            ("BACKGROUND FOLDER...", "background_settings", True, f"Open background-folder settings. Current mode: {background_mode}. Landscape images give the best result."),
            (f"SOUND: {'OFF' if self.sound.muted else 'ON'}", "sound", True, "Toggle generated game audio. Use [ and ] for volume."),
            ("FULLSCREEN", "fullscreen", True, "Toggle fullscreen (F11)."),
            ("QUIT", "quit", True, "Exit Frog-Vision."),
        ]
        self.buttons = []
        x, y, w, h, gap = 465, 190, 350, 39, 5
        for i, (label, action, enabled, tip) in enumerate(labels):
            self.buttons.append(Button(label, pygame.Rect(x, y + i * (h + gap), w, h), action, enabled, tip))
        self.selected_index = int(clamp(self.selected_index, 0, len(self.buttons) - 1))
        if self.buttons and not self.buttons[self.selected_index].enabled:
            self.select_next(1)

    def background_folder_buttons(self) -> list[Button]:
        has_custom = bool(str(self.settings.get("background_folder", "")).strip())
        return [
            Button("SELECT IMAGE FOLDER", pygame.Rect(440, 430, 400, 50), "background_select", True,
                   "Choose a folder containing PNG, JPG, BMP, or WEBP images."),
            Button("USE DEFAULT STARTUP FOLDER", pygame.Rect(440, 492, 400, 50), "background_default", has_custom,
                   "Return to FrameVision presets/startup."),
            Button("BACK", pygame.Rect(440, 554, 400, 48), "back"),
        ]

    def difficulty_buttons(self) -> list[Button]:
        return [
            Button(name.upper(), pygame.Rect(465, 275 + i * 70, 350, 52), f"diff:{name}", True,
                   f"{int(cfg['lives'])} lives · {int(cfg['timer'])} sec base timer · {cfg['score']:.2f}× score")
            for i, (name, cfg) in enumerate(DIFFICULTIES.items())
        ] + [Button("BACK", pygame.Rect(465, 500, 350, 48), "back")]

    def pause_buttons(self) -> list[Button]:
        return [
            Button("RESUME", pygame.Rect(470, 270, 340, 48), "resume"),
            Button("RESTART CAMPAIGN", pygame.Rect(470, 330, 340, 48), "restart_confirm"),
            Button("RETURN TO TITLE", pygame.Rect(470, 390, 340, 48), "title_confirm"),
        ]

    def end_buttons(self, victory: bool = False) -> list[Button]:
        return [
            Button("NEW GAME", pygame.Rect(470, 490, 340, 50), "new"),
            Button("STATISTICS", pygame.Rect(470, 550, 340, 50), "statistics"),
            Button("RETURN TO TITLE", pygame.Rect(470, 610, 340, 50), "title"),
        ]

    def select_next(self, direction: int) -> None:
        if not self.buttons:
            return
        start = self.selected_index
        while True:
            self.selected_index = (self.selected_index + direction) % len(self.buttons)
            if self.buttons[self.selected_index].enabled or self.selected_index == start:
                break
        self.sound.play("menu")

    # ------------------------------ state actions ----------------------------
    def perform_action(self, action: str) -> None:
        if action == "new":
            self.sound.play("confirm")
            self.new_game()
        elif action == "continue" and self.save.has_continue:
            self.sound.play("confirm")
            self.continue_game()
        elif action == "instructions":
            self.state = "instructions"
        elif action == "difficulty":
            self.state = "difficulty"
            self.buttons = self.difficulty_buttons()
            self.selected_index = list(DIFFICULTIES).index(self.settings["difficulty"])
        elif action == "background_settings":
            self.sound.play("confirm")
            self.state = "background_settings"
            self.buttons = self.background_folder_buttons()
            self.selected_index = 0
            self.background_notice = ""
        elif action == "background_select":
            self.choose_background_folder()
            self.buttons = self.background_folder_buttons()
        elif action == "background_default":
            self.use_default_background_folder()
            self.buttons = self.background_folder_buttons()
        elif action.startswith("diff:"):
            name = action.split(":", 1)[1]
            if name in DIFFICULTIES:
                self.settings["difficulty"] = name
                self.difficulty = name
                self.sound.play("confirm")
                self.write_settings()
                self.state = "title"
                self.build_title_buttons()
        elif action == "statistics":
            self.state = "statistics"
        elif action == "highscores":
            self.state = "highscores"
        elif action == "sound":
            self.sound.toggle_mute()
            self.write_settings()
            if self.state == "title":
                self.build_title_buttons()
        elif action == "fullscreen":
            self.toggle_fullscreen()
        elif action == "quit":
            self.open_confirm("quit", self.state)
        elif action == "back" or action == "title":
            self.state = "title"
            self.build_title_buttons()
        elif action == "resume":
            self.state = "playing"
            self.input.clear()
        elif action == "restart_confirm":
            self.open_confirm("restart", "paused")
        elif action == "title_confirm":
            self.open_confirm("title", "paused")
        elif action == "reset_save":
            self.open_confirm("reset_save", self.state)

    def open_confirm(self, action: str, return_state: str) -> None:
        self.confirm_action = action
        self.return_state = return_state
        self.state = "confirm"
        self.buttons = [
            Button("YES", pygame.Rect(450, 410, 170, 50), "confirm_yes"),
            Button("NO", pygame.Rect(660, 410, 170, 50), "confirm_no"),
        ]
        self.selected_index = 1

    def resolve_confirm(self, yes: bool) -> None:
        if not yes:
            self.state = self.return_state
            if self.state == "title":
                self.build_title_buttons()
            return
        action = self.confirm_action
        if action == "quit":
            self.running = False
        elif action == "restart":
            self.new_game()
        elif action == "title":
            self.save_campaign(True)
            self.state = "title"
            self.build_title_buttons()
        elif action == "reset_save":
            self.save.reset_all()
            self.settings = self.save.data["settings"]
            self.difficulty = self.settings["difficulty"]
            self.sound.volume = self.settings["volume"]
            self.sound.muted = self.settings["muted"]
            self.sound.apply_volume()
            self.background = self.load_background()
            self.state = "title"
            self.build_title_buttons()

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.write_settings()

    # ------------------------------ game mechanics ---------------------------
    def add_score(self, amount: int, x: Optional[float] = None, y: Optional[float] = None, label: Optional[str] = None) -> None:
        multiplier = 2 if self.effects.get("Score Multiplier", 0) > 0 else 1
        difficulty_mult = DIFFICULTIES[self.difficulty]["score"]
        gained = max(0, int(round(amount * multiplier * difficulty_mult)))
        self.score = max(0, self.score + gained)
        if x is not None and y is not None:
            self.floaters.append(FloatingText(label or f"+{gained}", x, y, YELLOW))
        stats = self.save.data["stats"]
        stats["high_score"] = max(stats["high_score"], self.score)
        while self.score >= self.next_life_score:
            self.lives += 1
            self.next_life_score += 20_000
            self.sound.play("life")
            self.floaters.append(FloatingText("EXTRA LIFE", DESIGN_W / 2, 150, LIME, 1.6, 1.6))

    def reset_attempt(self) -> None:
        self.player.reset()
        self.timer = self.timer_max
        self.best_row = START_ROW
        self.low_timer_tick = -1
        self.input.clear()
        if self.world:
            self.world.clear_spawn_area()

    def trigger_death(self, reason: str) -> None:
        if self.state != "playing" or self.player.invulnerable > 0:
            return
        if self.effects.get("Temporary Shield", 0) > 0:
            self.effects.pop("Temporary Shield", None)
            self.sound.play("bonus")
            self.screen_shake = .18
            self.flash = .18
            self.flash_color = CYAN
            self.spawn_burst(self.player.visual_rect.centerx, self.player.visual_rect.centery, CYAN, 18)
            self.floaters.append(FloatingText("SHIELD SAVED YOU", self.player.visual_rect.centerx, self.player.visual_rect.y, CYAN, 1.3, 1.3))
            self.reset_attempt()
            return
        self.state = "death"
        self.death_reason = reason
        self.death_timer = .95
        self.lives = max(0, self.lives - 1)
        self.deaths_this_level += 1
        self.campaign_deaths += 1
        self.crossing_streak = 0
        stats = self.save.data["stats"]
        stats["total_deaths"] += 1
        if reason == "VEHICLE IMPACT":
            stats["vehicle_collisions"] += 1
            self.sound.play("collision")
            color = RED
        elif reason in ("WATER", "CARRIED OFF", "UNSAFE PLATFORM", "MISSED FINISH"):
            stats["water_deaths"] += 1
            self.sound.play("splash")
            color = CYAN
        elif reason == "TIME EXPIRED":
            stats["timer_deaths"] += 1
            self.sound.play("collision")
            color = YELLOW
        else:
            stats["hazard_deaths"] += 1
            self.sound.play("collision")
            color = RED
        self.spawn_burst(self.player.visual_rect.centerx, self.player.visual_rect.centery, color, 32)
        self.screen_shake = .48
        self.flash = .23
        self.flash_color = color
        self.player.attached = None
        self.input.clear()
        self.save_campaign(True)

    def spawn_burst(self, x: float, y: float, color, count: int) -> None:
        for _ in range(count):
            angle = random.random() * math.tau
            speed = random.uniform(45, 230)
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed, random.uniform(.35, .9), .9, color, random.uniform(1.5, 4), 120))

    def road_collision(self) -> bool:
        if not self.world or self.player.invulnerable > 0 or self.player.row not in ROAD_ROWS:
            return False
        lane = self.world.road_lane_for_row(self.player.row)
        if lane is None:
            return False
        body = self.player.body_rect
        for v in lane.vehicles:
            swept = v.swept_rect.inflate(-7, -5)
            if body.colliderect(swept):
                self.trigger_death("VEHICLE IMPACT")
                return True
            trail = v.danger_trail
            if trail and body.colliderect(trail):
                self.trigger_death("VEHICLE IMPACT")
                return True
        return False

    @staticmethod
    def overlap_area(a: pygame.Rect, b: pygame.Rect) -> int:
        inter = a.clip(b)
        return max(0, inter.w) * max(0, inter.h)

    def check_river_support(self) -> bool:
        if not self.world or self.player.row not in RIVER_ROWS or self.player.hopping:
            return True
        lane = self.world.river_lane_for_row(self.player.row)
        if lane is None:
            self.trigger_death("WATER")
            return False
        stable = self.effects.get("Stable Platforms", 0) > 0
        feet = self.player.body_rect.inflate(8, 2)
        # Unsafe zones take precedence over support, especially a cyber-gator's head.
        for platform in lane.platforms:
            for danger in platform.unsafe_rects(stable):
                if self.overlap_area(feet, danger) > feet.w * feet.h * .23:
                    self.trigger_death("UNSAFE PLATFORM")
                    return False
        candidates: list[tuple[int, Platform]] = []
        for platform in lane.platforms:
            for support in platform.support_rects(stable):
                area = self.overlap_area(feet, support)
                if area >= feet.w * feet.h * .28:
                    candidates.append((area, platform))
        chosen: Optional[Platform] = None
        if self.player.attached:
            for area, platform in candidates:
                if platform is self.player.attached:
                    chosen = platform
                    break
        if chosen is None and candidates:
            candidates.sort(key=lambda item: (item[0], -abs(item[1].rect.centerx - feet.centerx)), reverse=True)
            chosen = candidates[0][1]
        if chosen is None:
            self.player.attached = None
            self.trigger_death("WATER")
            return False
        if self.player.attached is not chosen:
            self.sound.play("land")
        self.player.attached = chosen
        if self.player.body_rect.right < ARENA_X or self.player.body_rect.left > ARENA_X + ARENA_W:
            self.trigger_death("CARRIED OFF")
            return False
        return True

    def finish_slot_rects(self) -> list[pygame.Rect]:
        width = max(74, 112 - (self.level - 1) * 3)
        return [pygame.Rect(int(cx - width / 2), row_y(FINISH_ROW) + 5, width, LANE_H - 10) for cx in self.slot_centers]

    def handle_finish_landing(self) -> None:
        if self.state != "playing" or self.player.row != FINISH_ROW or self.player.hopping:
            return
        body = self.player.body_rect
        if self.world and any(body.colliderect(blocker) for blocker in self.world.finish_blockers()):
            self.trigger_death("FINISH BLOCKER")
            return
        slot_index = None
        for i, rect in enumerate(self.finish_slot_rects()):
            if rect.collidepoint(body.center) and self.overlap_area(body, rect) >= body.w * body.h * .50:
                slot_index = i
                break
        if slot_index is None:
            self.trigger_death("MISSED FINISH")
            return
        if slot_index in self.filled_slots:
            # Safe bounce-back avoids duplicate scoring and does not consume a life.
            self.sound.play("warning")
            self.floaters.append(FloatingText("SLOT ALREADY FILLED", self.slot_centers[slot_index], row_center(0) + 10, YELLOW, 1.2, 1.2))
            self.reset_attempt()
            return
        self.filled_slots.add(slot_index)
        self.crossing_streak += 1
        stats = self.save.data["stats"]
        stats["total_crossings"] += 1
        stats["successful_slots"] += 1
        time_bonus = int(self.timer * 15)
        base = 900 + self.level * 120 + self.crossing_streak * 75
        self.add_score(base + time_bonus, self.slot_centers[slot_index], row_center(0), "FINISH!")
        self.sound.play("finish")
        self.spawn_burst(self.slot_centers[slot_index], row_center(0), LIME, 28)
        self.flash = .16
        self.flash_color = LIME
        if self.world and self.world.bonus and self.world.bonus.kind == "Bonus Fly" and self.world.bonus.slot_index == slot_index:
            self.collect_bonus(self.world.bonus)
        if len(self.filled_slots) >= 5:
            self.complete_level()
        else:
            self.reset_attempt()
            if self.world:
                self.world.maybe_spawn_finish_fly(self.filled_slots, self.slot_centers)
            self.save_campaign(True)

    def collect_bonus(self, bonus: Bonus) -> None:
        if bonus.collected:
            return
        bonus.collected = True
        self.save.data["stats"]["bonuses_collected"] += 1
        kind = bonus.kind
        if kind == "Time Extension":
            self.timer = min(self.timer_max + 12, self.timer + 12)
        elif kind == "Extra Life":
            self.lives = min(99, self.lives + 1)
            self.sound.play("life")
        elif kind == "Score Multiplier":
            self.effects[kind] = max(self.effects.get(kind, 0), 10.0)
        elif kind == "Temporary Shield":
            self.effects[kind] = 16.0
        elif kind == "Slow Traffic":
            self.effects[kind] = max(self.effects.get(kind, 0), 9.0)
        elif kind == "Stable Platforms":
            self.effects[kind] = max(self.effects.get(kind, 0), 10.0)
        elif kind == "Bonus Fly":
            self.add_score(1250 + self.level * 100, bonus.x, row_center(bonus.row), "BONUS FLY")
        if kind != "Bonus Fly":
            self.add_score(350, bonus.x, row_center(bonus.row), kind.upper())
        self.sound.play("bonus")
        self.spawn_burst(bonus.x, row_center(bonus.row), YELLOW, 18)
        if self.world:
            self.world.bonus = None

    def check_bonus_collision(self) -> None:
        if not self.world or not self.world.bonus or self.world.bonus.kind == "Bonus Fly":
            return
        if self.player.body_rect.colliderect(self.world.bonus.rect) and self.player.row == self.world.bonus.row:
            self.collect_bonus(self.world.bonus)

    def check_hazards(self) -> None:
        if not self.world or self.player.invulnerable > 0:
            return
        body = self.player.body_rect
        for hazard in self.world.hazards:
            if hazard.kind == "Aerial Drone":
                if hazard.dangerous() and hazard.target and body.colliderect(hazard.rect):
                    self.trigger_death("AERIAL STRIKE")
                    return
            elif hazard.row == self.player.row and body.colliderect(hazard.rect.inflate(-8, -5)):
                self.trigger_death(hazard.kind.upper())
                return

    def complete_level(self) -> None:
        if self.state != "playing" or self.level_bonus_awarded:
            return
        self.level_bonus_awarded = True
        flawless = self.deaths_this_level == 0
        level_bonus = 2500 + self.level * 500 + (1800 if flawless else 0)
        self.add_score(level_bonus)
        stats = self.save.data["stats"]
        stats["levels_completed"] += 1
        stats["highest_level"] = max(stats["highest_level"], self.level)
        best = stats["best_level_time"]
        stats["best_level_time"] = self.level_elapsed if best is None else min(best, self.level_elapsed)
        if self.level % 3 == 0:
            self.lives += 1
            self.sound.play("life")
        self.level_summary = {
            "level": self.level,
            "bonus": level_bonus,
            "flawless": flawless,
            "time": self.level_elapsed,
        }
        self.state = "level_complete"
        self.sound.play("level")
        if self.level >= MAX_LEVEL:
            if not self.game_result_recorded:
                self.save.data["stats"]["games_won"] += 1
                self.game_result_recorded = True
            self.save.data["campaign"] = {"active": False}
            self.write_settings()
        else:
            self.save_campaign(True)

    def advance_level(self) -> None:
        if self.level >= MAX_LEVEL:
            self.win_game()
            return
        self.level += 1
        self.deaths_this_level = 0
        self.filled_slots.clear()
        self.start_level(intro=True)
        self.save_campaign(True)

    def lose_game(self) -> None:
        if not self.game_result_recorded:
            self.save.data["stats"]["games_lost"] += 1
            self.game_result_recorded = True
        self.save.data["campaign"] = {"active": False}
        self.save.save()
        self.state = "game_over"
        self.sound.play("gameover")

    def win_game(self) -> None:
        if not self.game_result_recorded:
            self.save.data["stats"]["games_won"] += 1
            self.game_result_recorded = True
        self.save.data["campaign"] = {"active": False}
        self.save.data["stats"]["high_score"] = max(self.save.data["stats"]["high_score"], self.score)
        self.save.save()
        self.state = "victory"
        self.sound.play("victory")
        self.spawn_burst(DESIGN_W / 2, DESIGN_H / 2, CYAN, 90)

    # ------------------------------- event loop ------------------------------
    def current_buttons(self) -> list[Button]:
        if self.state == "title":
            self.build_title_buttons()
        elif self.state == "difficulty":
            self.buttons = self.difficulty_buttons()
        elif self.state == "background_settings":
            self.buttons = self.background_folder_buttons()
        elif self.state == "paused":
            self.buttons = self.pause_buttons()
        elif self.state in ("game_over", "victory"):
            self.buttons = self.end_buttons(self.state == "victory")
        elif self.state == "statistics":
            self.buttons = [
                Button("RESET SAVE DATA", pygame.Rect(455, 590, 370, 48), "reset_save", True, "Erase settings, campaign progress, high score, and all statistics."),
                Button("BACK", pygame.Rect(455, 648, 370, 44), "back"),
            ]
        elif self.state == "highscores":
            self.buttons = [Button("BACK", pygame.Rect(470, 610, 340, 48), "back")]
        return self.buttons

    def design_mouse(self, pos: tuple[int, int]) -> tuple[int, int]:
        sw, sh = self.screen.get_size()
        scale = min(sw / DESIGN_W, sh / DESIGN_H)
        dw, dh = DESIGN_W * scale, DESIGN_H * scale
        ox, oy = (sw - dw) / 2, (sh - dh) / 2
        if scale <= 0:
            return -999, -999
        return int((pos[0] - ox) / scale), int((pos[1] - oy) / scale)

    def activate_selected(self) -> None:
        buttons = self.current_buttons()
        if not buttons:
            return
        self.selected_index = int(clamp(self.selected_index, 0, len(buttons) - 1))
        button = buttons[self.selected_index]
        if not button.enabled:
            return
        if button.action == "confirm_yes":
            self.resolve_confirm(True)
        elif button.action == "confirm_no":
            self.resolve_confirm(False)
        else:
            self.perform_action(button.action)

    def activate_button(self, button: Button) -> None:
        """Execute the exact button that received the mouse click.

        Mouse activation must not depend on selected_index because title-menu
        buttons are rebuilt dynamically. Direct dispatch keeps the clicked
        action stable even when Continue Game appears or disappears.
        """
        if not button.enabled:
            return
        if button.action == "confirm_yes":
            self.resolve_confirm(True)
        elif button.action == "confirm_no":
            self.resolve_confirm(False)
        else:
            self.perform_action(button.action)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.state == "confirm" and self.confirm_action == "quit":
                    self.running = False
                else:
                    self.open_confirm("quit", self.state)
                continue
            if event.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.windowed_size = (max(320, event.w), max(180, event.h))
                self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE | pygame.DOUBLEBUF)
                continue
            if event.type == pygame.WINDOWFOCUSLOST:
                self.input.clear()
                if self.state == "playing":
                    self.state = "paused"
                    self.buttons = self.pause_buttons()
                    self.selected_index = 0
                continue
            if event.type == pygame.KEYUP:
                self.input.key_up(event.key)
                continue
            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_F11:
                    self.toggle_fullscreen()
                    continue
                if key == pygame.K_m:
                    self.sound.toggle_mute()
                    self.write_settings()
                    if self.state == "title":
                        self.build_title_buttons()
                    continue
                if key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    self.sound.change_volume(-.08)
                    self.write_settings()
                    continue
                if key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                    self.sound.change_volume(.08)
                    self.write_settings()
                    continue

                if self.state == "playing":
                    if key in (pygame.K_p, pygame.K_ESCAPE):
                        self.state = "paused"
                        self.buttons = self.pause_buttons()
                        self.selected_index = 0
                        self.input.clear()
                    else:
                        self.input.key_down(key)
                elif self.state == "paused":
                    if key in (pygame.K_p, pygame.K_ESCAPE):
                        self.state = "playing"
                        self.input.clear()
                    elif key in (pygame.K_UP, pygame.K_w):
                        self.select_next(-1)
                    elif key in (pygame.K_DOWN, pygame.K_s):
                        self.select_next(1)
                    elif key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.activate_selected()
                elif self.state in ("title", "difficulty", "background_settings", "confirm", "game_over", "victory", "statistics", "highscores"):
                    buttons = self.current_buttons()
                    if self.state == "title" and key == pygame.K_b:
                        self.perform_action("background_settings")
                    elif key in (pygame.K_UP, pygame.K_w):
                        self.select_next(-1)
                    elif key in (pygame.K_DOWN, pygame.K_s):
                        self.select_next(1)
                    elif key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.activate_selected()
                    elif key == pygame.K_ESCAPE:
                        if self.state == "confirm":
                            self.resolve_confirm(False)
                        elif self.state in ("difficulty", "background_settings", "statistics", "highscores"):
                            self.state = "title"
                            self.build_title_buttons()
                        elif self.state in ("game_over", "victory"):
                            self.state = "title"
                            self.build_title_buttons()
                        elif self.state == "title":
                            self.open_confirm("quit", "title")
                    elif self.state == "game_over" and key == pygame.K_r:
                        self.new_game()
                elif self.state == "instructions":
                    if key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                        self.state = "title"
                        self.build_title_buttons()
                elif self.state == "level_intro":
                    if key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.state = "playing"
                        self.input.clear()
                    elif key == pygame.K_ESCAPE:
                        self.open_confirm("title", "level_intro")
                elif self.state == "level_complete":
                    if key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.advance_level()
                    elif key == pygame.K_ESCAPE:
                        self.open_confirm("title", "level_complete")
                elif self.state == "death":
                    pass

            if event.type == pygame.MOUSEMOTION:
                self.mouse_design = self.design_mouse(event.pos)
                buttons = self.current_buttons() if self.state in ("title", "difficulty", "background_settings", "paused", "confirm", "game_over", "victory", "statistics", "highscores") else []
                for i, button in enumerate(buttons):
                    if button.enabled and button.rect.collidepoint(self.mouse_design):
                        self.selected_index = i
                        break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.mouse_design = self.design_mouse(event.pos)
                if self.state == "level_intro":
                    self.state = "playing"
                    self.input.clear()
                    continue
                if self.state == "level_complete":
                    self.advance_level()
                    continue
                if self.state == "instructions":
                    self.state = "title"
                    self.build_title_buttons()
                    continue
                buttons = self.current_buttons() if self.state in ("title", "difficulty", "background_settings", "paused", "confirm", "game_over", "victory", "statistics", "highscores") else []
                for i, button in enumerate(buttons):
                    if button.enabled and button.rect.collidepoint(self.mouse_design):
                        self.selected_index = i
                        self.activate_button(button)
                        break

    def update_ambient(self, dt: float) -> None:
        for p in self.ambient:
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.y < -8:
                p.y = DESIGN_H + 8
                p.x = random.uniform(0, DESIGN_W)
            if p.x < -8:
                p.x = DESIGN_W + 8
            elif p.x > DESIGN_W + 8:
                p.x = -8
        self.particles = [p for p in self.particles if p.update(dt)]
        self.floaters = [f for f in self.floaters if f.update(dt)]
        self.screen_shake = max(0.0, self.screen_shake - dt)
        self.flash = max(0.0, self.flash - dt)
        self.message_timer = max(0.0, self.message_timer - dt)

    def update_playing(self, dt: float) -> None:
        if not self.world:
            return
        self.input.update(dt)
        if not self.player.hopping:
            move = self.input.pop_move()
            if move and self.player.try_hop(*move):
                self.sound.play("hop")

        old_warning = self.world.special_warning > 0
        self.world.update(dt, self.player, self.effects)
        if self.world.special_warning > 0 and not old_warning:
            self.sound.play("warning")

        if self.player.attached and not self.player.hopping:
            self.player.carry(self.player.attached.dx)
        landed = self.player.update(dt)

        if landed and self.player.row < self.best_row and self.player.row != FINISH_ROW:
            advanced = self.best_row - self.player.row
            self.best_row = self.player.row
            self.add_score(70 * advanced + self.level * 5, self.player.visual_rect.centerx, self.player.visual_rect.y, "FORWARD")

        if self.state != "playing":
            return
        if self.road_collision():
            return
        self.check_hazards()
        if self.state != "playing":
            return
        if not self.player.hopping and self.player.row in RIVER_ROWS:
            self.check_river_support()
        elif self.player.row not in RIVER_ROWS:
            self.player.attached = None
        if self.state != "playing":
            return
        if landed and self.player.row == FINISH_ROW:
            self.handle_finish_landing()
        if self.state != "playing":
            return
        self.check_bonus_collision()

        # A restrained vehicle pass sound: crossing the player's x on an adjacent lane can only fire once per vehicle pass.
        for lane in self.world.road_lanes:
            for v in lane.vehicles:
                v.pass_cooldown = max(0.0, v.pass_cooldown - dt)
                if abs(lane.row - self.player.row) <= 1 and abs(v.rect.centerx - self.player.body_rect.centerx) < 5 and v.pass_cooldown <= 0:
                    self.sound.play("pass")
                    v.pass_cooldown = 1.2

        for key in list(self.effects):
            self.effects[key] -= dt
            if self.effects[key] <= 0:
                del self.effects[key]
        self.level_elapsed += dt
        self.timer = max(0.0, self.timer - dt)
        if self.timer <= 8.0:
            sec = int(math.ceil(self.timer))
            if sec != self.low_timer_tick:
                self.low_timer_tick = sec
                self.sound.play("tick")
        if self.timer <= 0:
            self.trigger_death("TIME EXPIRED")

    def update(self, dt: float) -> None:
        dt = min(dt, .05)
        self.update_ambient(dt)
        if self.state == "playing":
            self.update_playing(dt)
        elif self.state == "death":
            self.death_timer -= dt
            if self.death_timer <= 0:
                if self.lives <= 0:
                    self.lose_game()
                else:
                    self.reset_attempt()
                    self.state = "playing"
        now = time.monotonic()
        if now - self.last_stats_save > 20.0:
            self.flush_play_time()
            self.save.save()
            self.last_stats_save = now

    # -------------------------------- drawing --------------------------------
    def draw_background(self, surface: pygame.Surface) -> None:
        surface.blit(self.background, (0, 0))
        dark_alpha = 145
        if self.world and self.world.special_name == "Night Crossing" and self.state not in ("title", "instructions", "statistics", "highscores"):
            dark_alpha = 188
        surface.blit(pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA), (0, 0))
        shade = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        shade.fill((2, 6, 15, dark_alpha))
        surface.blit(shade, (0, 0))
        t = pygame.time.get_ticks() / 1000.0
        grid = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        for x in range(-60, DESIGN_W + 60, 48):
            px = int(x + (t * 12) % 48)
            pygame.draw.line(grid, (20, 180, 210, 22), (px, 0), (px, DESIGN_H), 1)
        for y in range(0, DESIGN_H, 48):
            pygame.draw.line(grid, (20, 130, 180, 18), (0, y), (DESIGN_W, y), 1)
        surface.blit(grid, (0, 0))
        for p in self.ambient:
            pygame.draw.circle(surface, (*p.color, 65), (int(p.x), int(p.y)), int(p.size))

    def draw_playfield_base(self, surface: pygame.Surface) -> None:
        lanes = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)
        t = pygame.time.get_ticks() / 1000.0
        # Finish / grass banks.
        for row in (FINISH_ROW, MEDIAN_ROW, START_ROW):
            local_y = row * LANE_H
            pygame.draw.rect(lanes, (*GRASS, 185), (0, local_y, ARENA_W, LANE_H))
            for x in range(0, ARENA_W, 32):
                h = 3 + int(2 * math.sin(t * 2 + x * .05 + row))
                pygame.draw.line(lanes, (*TEAL, 90), (x, local_y + LANE_H - 5), (x + 5, local_y + LANE_H - 5 - h), 1)
        # River with animated wave lines.
        for row in RIVER_ROWS:
            local_y = row * LANE_H
            pygame.draw.rect(lanes, (*WATER, 182), (0, local_y, ARENA_W, LANE_H))
            for x in range(-20, ARENA_W + 20, 52):
                wave_x = int(x + (t * (20 + row * 3) * (-1 if row % 2 else 1)) % 52)
                pygame.draw.arc(lanes, (20, 155, 220, 75), (wave_x, local_y + 8, 35, 16), 0, math.pi, 2)
                pygame.draw.arc(lanes, (25, 210, 220, 45), (wave_x + 18, local_y + 26, 28, 11), math.pi, math.tau, 1)
        # Roads and lane markers.
        for row in ROAD_ROWS:
            local_y = row * LANE_H
            pygame.draw.rect(lanes, (*ROAD, 205), (0, local_y, ARENA_W, LANE_H))
            for x in range(-80, ARENA_W + 80, 112):
                offset = int((t * 18 * (1 if row % 2 else -1)) % 112)
                pygame.draw.rect(lanes, (105, 180, 205, 95), (x + offset, local_y + LANE_H // 2 - 2, 58, 4), border_radius=2)
        for row in range(ROWS + 1):
            pygame.draw.line(lanes, (30, 165, 205, 70), (0, row * LANE_H), (ARENA_W, row * LANE_H), 1)
        surface.blit(lanes, (ARENA_X, ARENA_Y))
        draw_glow(surface, pygame.Rect(ARENA_X, ARENA_Y, ARENA_W, ARENA_H), CYAN, 7, 2)

    def draw_finish_area(self, surface: pygame.Surface) -> None:
        filled = self.filled_slots
        for i, rect in enumerate(self.finish_slot_rects()):
            col = LIME if i in filled else CYAN
            pygame.draw.rect(surface, (4, 16, 25), rect, border_radius=13)
            draw_glow(surface, rect, col, 6, 2)
            pulse = 3 + int(2 * math.sin(pygame.time.get_ticks() * .006 + i))
            pygame.draw.circle(surface, (*col, 120), rect.center, 10 + pulse, 2)
            if i in filled:
                pygame.draw.ellipse(surface, LIME, pygame.Rect(rect.centerx - 12, rect.centery - 9, 24, 18))
                pygame.draw.circle(surface, WHITE, (rect.centerx - 6, rect.centery - 6), 3)
                pygame.draw.circle(surface, WHITE, (rect.centerx + 6, rect.centery - 6), 3)
        if self.world:
            for blocker in self.world.finish_blockers():
                draw_glow(surface, blocker, RED, 7, 2)
                pygame.draw.rect(surface, (70, 26, 35), blocker, border_radius=7)
                pygame.draw.line(surface, YELLOW, blocker.topleft, blocker.bottomright, 3)
                pygame.draw.line(surface, YELLOW, blocker.topright, blocker.bottomleft, 3)

    def draw_hud(self, surface: pygame.Surface) -> None:
        stats = self.save.data["stats"]
        pygame.draw.rect(surface, (6, 12, 24, 235), (0, 0, DESIGN_W, 82))
        pygame.draw.line(surface, CYAN, (0, 81), (DESIGN_W, 81), 2)
        title = self.fonts["menu"].render("FROG-VISION", True, CYAN)
        surface.blit(title, (24, 13))
        info = self.fonts["small"]
        items = [
            (f"SCORE {self.score:08d}", 245),
            (f"HIGH {max(stats['high_score'], self.score):08d}", 455),
            (f"LIVES {self.lives}", 685),
            (f"LEVEL {self.level}/{MAX_LEVEL}", 810),
            (self.difficulty.upper(), 950),
            ("MUTED" if self.sound.muted else f"VOL {int(self.sound.volume * 100):02d}", 1110),
        ]
        for text, x in items:
            surface.blit(info.render(text, True, WHITE), (x, 18))
        slot_text = f"SLOTS {len(self.filled_slots)}/5"
        surface.blit(info.render(slot_text, True, LIME), (24, 51))
        # Time bar and numeric timer.
        bar = pygame.Rect(245, 52, 560, 18)
        pygame.draw.rect(surface, (22, 31, 45), bar, border_radius=8)
        ratio = clamp(self.timer / max(.001, self.timer_max), 0, 1)
        time_col = LIME if ratio > .45 else YELLOW if ratio > .20 else RED
        fill = bar.copy(); fill.w = int(bar.w * ratio)
        pygame.draw.rect(surface, time_col, fill, border_radius=8)
        pygame.draw.rect(surface, WHITE, bar, 1, border_radius=8)
        surface.blit(info.render(f"TIME {self.timer:04.1f}", True, time_col), (820, 48))
        effects = [f"{k}: {v:.0f}s" for k, v in self.effects.items()]
        if effects:
            text = "  ·  ".join(effects[:3])
            surface.blit(self.fonts["tiny"].render(text, True, YELLOW), (965, 52))

    def draw_special_effects(self, surface: pygame.Surface) -> None:
        if not self.world:
            return
        if self.world.special_name == "Storm River":
            rain = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            t = pygame.time.get_ticks()
            for i in range(90):
                x = (i * 97 + t // 8) % DESIGN_W
                y = (i * 53 + t // 3) % DESIGN_H
                pygame.draw.line(rain, (150, 220, 255, 75), (x, y), (x - 9, y + 22), 1)
            lightning = (self.world.special_clock % 6.8) < .16
            if lightning:
                rain.fill((180, 225, 255, 45), special_flags=pygame.BLEND_RGBA_ADD)
            else:
                pygame.draw.rect(rain, (0, 8, 24, 42), (ARENA_X, row_y(1), ARENA_W, LANE_H * len(RIVER_ROWS)))
            surface.blit(rain, (0, 0))
        if self.world.special_name == "Reversing Current" and self.world.reversal_lanes:
            for idx in self.world.reversal_lanes:
                lane = self.world.river_lanes[idx]
                rr = pygame.Rect(ARENA_X + 2, row_y(lane.row) + 2, ARENA_W - 4, LANE_H - 4)
                pygame.draw.rect(surface, YELLOW, rr, 3, border_radius=5)
                arrow = "<<< CURRENT REVERSAL <<<" if lane.direction > 0 else ">>> CURRENT REVERSAL >>>"
                img = self.fonts["tiny"].render(arrow, True, YELLOW)
                surface.blit(img, img.get_rect(center=rr.center))
        if self.world.special_warning > 0:
            alpha = int(55 + 90 * self.world.special_warning)
            warning = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)
            warning.fill((*YELLOW, alpha))
            surface.blit(warning, (ARENA_X, ARENA_Y), special_flags=pygame.BLEND_RGBA_ADD)
            text = self.fonts["body"].render("⚠  SPEED CHANGE INCOMING", True, YELLOW)
            surface.blit(text, text.get_rect(center=(DESIGN_W // 2, 106)))
        if self.world.special_active:
            text = self.fonts["small"].render(self.world.special_name.upper() + " ACTIVE", True, RED if "Hour" in self.world.special_name else CYAN)
            surface.blit(text, text.get_rect(center=(DESIGN_W // 2, DESIGN_H - 16)))

    def draw_game(self, surface: pygame.Surface) -> None:
        self.draw_playfield_base(surface)
        self.draw_finish_area(surface)
        if self.world:
            self.world.draw_objects(surface, self.effects)
            if self.world.bonus:
                self.world.bonus.draw(surface, self.fonts["tiny"])
        self.player.draw(surface, self.effects.get("Temporary Shield", 0) > 0)
        fx = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        for p in self.particles:
            p.draw(fx)
        surface.blit(fx, (0, 0))
        for f in self.floaters:
            f.draw(surface, self.fonts["small"])
        self.draw_special_effects(surface)
        self.draw_hud(surface)
        if self.state == "death":
            msg = self.fonts["huge"].render(self.death_reason, True, self.flash_color)
            surface.blit(msg, msg.get_rect(center=(DESIGN_W // 2, DESIGN_H // 2)))

    def panel(self, surface: pygame.Surface, rect: pygame.Rect, title: str) -> None:
        pygame.draw.rect(surface, (*PANEL, 235), rect, border_radius=18)
        draw_glow(surface, rect, CYAN, 10, 2)
        text = self.fonts["menu"].render(title, True, CYAN)
        surface.blit(text, text.get_rect(midtop=(rect.centerx, rect.y + 18)))

    def draw_buttons(self, surface: pygame.Surface) -> None:
        buttons = self.current_buttons()
        for i, button in enumerate(buttons):
            button.draw(surface, self.fonts["body"], self.mouse_design, i == self.selected_index)
        if buttons:
            selected = buttons[int(clamp(self.selected_index, 0, len(buttons) - 1))]
            tip = selected.tooltip
            for button in buttons:
                if button.enabled and button.rect.collidepoint(self.mouse_design):
                    tip = button.tooltip
            if tip:
                text = self.fonts["tiny"].render(tip, True, (185, 220, 235))
                surface.blit(text, text.get_rect(center=(DESIGN_W // 2, 686)))

    def draw_title(self, surface: pygame.Surface) -> None:
        title = self.fonts["title"].render("FROG-VISION", True, CYAN)
        surface.blit(title, title.get_rect(center=(DESIGN_W // 2, 78)))
        sub = self.fonts["small"].render("NEON CROSSING PROTOCOL // TEN-LEVEL ARCADE CAMPAIGN", True, LIME)
        surface.blit(sub, sub.get_rect(center=(DESIGN_W // 2, 128)))
        self.draw_buttons(surface)
        src = self.fonts["tiny"].render(f"Background: {self.background_name} ({self.background_source})   ·   B background folder   ·   F11 fullscreen   ·   M mute", True, (135, 178, 195))
        surface.blit(src, src.get_rect(center=(DESIGN_W // 2, 704)))

    def draw_instructions(self, surface: pygame.Surface) -> None:
        rect = pygame.Rect(110, 70, 1060, 590)
        self.panel(surface, rect, "INSTRUCTIONS")
        lines = [
            ("MOVE", "Arrow keys or WASD. One press makes one lane-sized hop; holding repeats slowly and deliberately."),
            ("GOAL", "Cross five road lanes, rest on the median, ride five river lanes, and fill all five finish slots."),
            ("ROAD", "Cars, sports cars, trucks, buses, motorcycles, and hazard transports use swept collision checks."),
            ("RIVER", "Land on logs, trunks, turtles, rafts, broken logs, or the safe back of a cyber-gator."),
            ("WARNINGS", "Turtles sink, broken logs rotate, gators open their mouths, currents reverse, and traffic surges."),
            ("BONUSES", "+Time, extra life, 2× score, shield, slow traffic, stable platforms, and finish-slot bonus flies."),
            ("SCORING", "First progress into each higher row scores. Finish slots, remaining time, streaks, and flawless levels add more."),
            ("SYSTEM", "P/Esc pause · R restarts after game over · M mute · [ / ] volume · F11 fullscreen."),
            ("SAVE", "Campaign progress, settings, high score, and lifetime statistics save automatically in FrameVision presets/setsave."),
        ]
        y = 132
        for head, body in lines:
            surface.blit(self.fonts["small"].render(head, True, YELLOW), (145, y))
            wrapped = self.wrap_text(body, self.fonts["small"], 830)
            for j, line in enumerate(wrapped):
                surface.blit(self.fonts["small"].render(line, True, WHITE), (300, y + j * 24))
            y += max(48, 26 * len(wrapped))
        hint = self.fonts["body"].render("Press Enter, Space, Esc, or click to return", True, CYAN)
        surface.blit(hint, hint.get_rect(center=(DESIGN_W // 2, 635)))

    @staticmethod
    def wrap_text(text: str, font: pygame.font.Font, width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = word if not current else current + " " + word
            if font.size(trial)[0] <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def draw_background_settings(self, surface: pygame.Surface) -> None:
        self.panel(surface, pygame.Rect(230, 85, 820, 555), "BACKGROUND IMAGES")
        custom = self.custom_background_dir()
        configured = custom if custom is not None else self.startup_dir
        mode = "CUSTOM FOLDER" if custom is not None else "DEFAULT FRAMEVISION FOLDER"
        mode_img = self.fonts["body"].render(mode, True, YELLOW if custom is not None else CYAN)
        surface.blit(mode_img, mode_img.get_rect(center=(DESIGN_W // 2, 165)))

        tip = self.fonts["small"].render("Tip: landscape images give the best result with the least cropping.", True, LIME)
        surface.blit(tip, tip.get_rect(center=(DESIGN_W // 2, 210)))
        info = self.fonts["small"].render("A new random image is loaded automatically for every level.", True, WHITE)
        surface.blit(info, info.get_rect(center=(DESIGN_W // 2, 245)))

        path_title = self.fonts["tiny"].render("CONFIGURED FOLDER", True, (135, 190, 210))
        surface.blit(path_title, path_title.get_rect(center=(DESIGN_W // 2, 292)))
        path_lines = self.wrap_text(str(configured), self.fonts["small"], 720)
        for i, line in enumerate(path_lines[:3]):
            path_img = self.fonts["small"].render(line, True, WHITE)
            surface.blit(path_img, path_img.get_rect(center=(DESIGN_W // 2, 322 + i * 25)))

        source_text = f"Current image: {self.background_name} ({self.background_source.lower()} source)"
        source_img = self.fonts["tiny"].render(source_text, True, (150, 205, 220))
        surface.blit(source_img, source_img.get_rect(center=(DESIGN_W // 2, 390)))
        if self.background_notice:
            notice_lines = self.wrap_text(self.background_notice, self.fonts["tiny"], 760)
            for i, line in enumerate(notice_lines[:2]):
                notice = self.fonts["tiny"].render(line, True, YELLOW)
                surface.blit(notice, notice.get_rect(center=(DESIGN_W // 2, 412 + i * 18)))
        self.draw_buttons(surface)

    def draw_difficulty(self, surface: pygame.Surface) -> None:
        self.panel(surface, pygame.Rect(390, 150, 500, 430), "SELECT DIFFICULTY")
        self.draw_buttons(surface)

    def draw_statistics(self, surface: pygame.Surface) -> None:
        self.panel(surface, pygame.Rect(220, 55, 840, 650), "STATISTICS")
        s = self.save.data["stats"]
        best = "—" if s["best_level_time"] is None else f"{s['best_level_time']:.1f} sec"
        hours = s["total_play_time"] / 3600
        rows = [
            ("High score", f"{s['high_score']:,}"), ("Highest level", str(s["highest_level"])),
            ("Total crossings", f"{s['total_crossings']:,}"), ("Successful finish slots", f"{s['successful_slots']:,}"),
            ("Total deaths", f"{s['total_deaths']:,}"), ("Vehicle collisions", f"{s['vehicle_collisions']:,}"),
            ("Water deaths", f"{s['water_deaths']:,}"), ("Timer deaths", f"{s['timer_deaths']:,}"),
            ("Hazard deaths", f"{s['hazard_deaths']:,}"), ("Bonuses collected", f"{s['bonuses_collected']:,}"),
            ("Levels completed", f"{s['levels_completed']:,}"), ("Games won / lost", f"{s['games_won']} / {s['games_lost']}"),
            ("Total play time", f"{hours:.2f} hours"), ("Best level completion", best),
        ]
        for i, (label, value) in enumerate(rows):
            col = i % 2
            row = i // 2
            x = 270 + col * 390
            y = 130 + row * 56
            surface.blit(self.fonts["small"].render(label.upper(), True, (135, 190, 210)), (x, y))
            surface.blit(self.fonts["body"].render(value, True, WHITE), (x, y + 22))
        self.draw_buttons(surface)

    def draw_highscores(self, surface: pygame.Surface) -> None:
        self.panel(surface, pygame.Rect(310, 90, 660, 570), "HIGH SCORE")
        s = self.save.data["stats"]
        score = self.fonts["huge"].render(f"{s['high_score']:,}", True, YELLOW)
        surface.blit(score, score.get_rect(center=(DESIGN_W // 2, 235)))
        details = [
            f"Highest level reached: {s['highest_level']}",
            f"Campaign victories: {s['games_won']}",
            f"Levels completed: {s['levels_completed']}",
            f"Successful slot landings: {s['successful_slots']}",
            "A new high score is saved immediately and survives campaign restarts.",
        ]
        y = 330
        for line in details:
            img = self.fonts["body"].render(line, True, WHITE if not line.startswith("A new") else CYAN)
            surface.blit(img, img.get_rect(center=(DESIGN_W // 2, y)))
            y += 48
        self.draw_buttons(surface)

    def draw_level_intro(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA); veil.fill((2, 6, 15, 185)); surface.blit(veil, (0, 0))
        self.panel(surface, pygame.Rect(250, 155, 780, 405), f"LEVEL {self.level} // {self.world.special_name if self.world else ''}")
        desc = self.world.special_desc if self.world else ""
        y = 250
        for line in self.wrap_text(desc, self.fonts["menu"], 650):
            img = self.fonts["menu"].render(line, True, WHITE)
            surface.blit(img, img.get_rect(center=(DESIGN_W // 2, y)))
            y += 40
        info = [
            f"Difficulty: {self.difficulty}   ·   Lives: {self.lives}   ·   Attempt timer: {self.timer_max:.1f}s",
            f"Filled slots restored: {len(self.filled_slots)}/5",
            "Press Enter, Space, or click to begin",
        ]
        for i, line in enumerate(info):
            col = CYAN if i == len(info) - 1 else (160, 205, 220)
            img = self.fonts["body"].render(line, True, col)
            surface.blit(img, img.get_rect(center=(DESIGN_W // 2, 395 + i * 48)))

    def draw_pause(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA); veil.fill((1, 4, 10, 195)); surface.blit(veil, (0, 0))
        self.panel(surface, pygame.Rect(400, 190, 480, 310), "PAUSED")
        self.draw_buttons(surface)

    def draw_level_complete(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA); veil.fill((1, 5, 12, 205)); surface.blit(veil, (0, 0))
        self.panel(surface, pygame.Rect(300, 130, 680, 470), "LEVEL COMPLETE")
        rows = [
            f"Level {self.level} cleared",
            f"Level time: {self.level_summary.get('time', 0):.1f} seconds",
            f"Completion bonus: {self.level_summary.get('bonus', 0):,}",
            "Flawless crossing bonus earned" if self.level_summary.get("flawless") else f"Deaths this level: {self.deaths_this_level}",
            f"Current score: {self.score:,}",
            "Final level cleared — continue to victory" if self.level >= MAX_LEVEL else f"Next: Level {self.level + 1}",
        ]
        for i, line in enumerate(rows):
            img = self.fonts["menu" if i in (0, 4) else "body"].render(line, True, YELLOW if i == 4 else WHITE)
            surface.blit(img, img.get_rect(center=(DESIGN_W // 2, 225 + i * 52)))
        hint = self.fonts["body"].render("Press Enter, Space, or click", True, CYAN)
        surface.blit(hint, hint.get_rect(center=(DESIGN_W // 2, 555)))

    def draw_end(self, surface: pygame.Surface, victory: bool) -> None:
        veil = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA); veil.fill((1, 4, 10, 205)); surface.blit(veil, (0, 0))
        title = "FINAL VICTORY" if victory else "GAME OVER"
        color = LIME if victory else RED
        img = self.fonts["huge"].render(title, True, color)
        surface.blit(img, img.get_rect(center=(DESIGN_W // 2, 125)))
        stats = self.save.data["stats"]
        summary = [
            f"Score: {self.score:,}", f"High score: {stats['high_score']:,}",
            f"Level reached: {self.level}", f"Campaign deaths: {self.campaign_deaths:,}",
            (f"Best level time: {stats['best_level_time']:.1f}s" if stats["best_level_time"] is not None else "Best level time: —"),
            "All ten crossing sectors secured." if victory else "The crossing network remains active.",
        ]
        for i, line in enumerate(summary):
            text = self.fonts["menu" if i < 2 else "body"].render(line, True, YELLOW if i == 0 else WHITE)
            surface.blit(text, text.get_rect(center=(DESIGN_W // 2, 210 + i * 46)))
        self.draw_buttons(surface)

    def draw_confirm(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA); veil.fill((0, 0, 0, 210)); surface.blit(veil, (0, 0))
        self.panel(surface, pygame.Rect(340, 245, 600, 260), "CONFIRM")
        messages = {
            "quit": "Quit Frog-Vision? Current safe campaign progress will be saved.",
            "restart": "Restart the campaign from level 1? Long-term statistics stay saved.",
            "title": "Return to the title screen? Current safe progress will be saved.",
            "reset_save": "Erase ALL saved progress, settings, high score, and statistics?",
        }
        msg = messages.get(self.confirm_action, "Continue?")
        for i, line in enumerate(self.wrap_text(msg, self.fonts["body"], 500)):
            text = self.fonts["body"].render(line, True, WHITE)
            surface.blit(text, text.get_rect(center=(DESIGN_W // 2, 330 + i * 32)))
        self.draw_buttons(surface)

    def render(self) -> None:
        self.canvas.fill(BLACK)
        self.draw_background(self.canvas)
        if self.state in ("playing", "paused", "death", "level_intro", "level_complete", "game_over", "victory"):
            self.draw_game(self.canvas)
        if self.state == "title": self.draw_title(self.canvas)
        elif self.state == "instructions": self.draw_instructions(self.canvas)
        elif self.state == "difficulty": self.draw_difficulty(self.canvas)
        elif self.state == "background_settings": self.draw_background_settings(self.canvas)
        elif self.state == "statistics": self.draw_statistics(self.canvas)
        elif self.state == "highscores": self.draw_highscores(self.canvas)
        elif self.state == "level_intro": self.draw_level_intro(self.canvas)
        elif self.state == "paused": self.draw_pause(self.canvas)
        elif self.state == "level_complete": self.draw_level_complete(self.canvas)
        elif self.state == "game_over": self.draw_end(self.canvas, False)
        elif self.state == "victory": self.draw_end(self.canvas, True)
        elif self.state == "confirm": self.draw_confirm(self.canvas)

        if self.flash > 0:
            flash = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            flash.fill((*self.flash_color, int(110 * self.flash / .23)))
            self.canvas.blit(flash, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        sw, sh = self.screen.get_size()
        scale = min(sw / DESIGN_W, sh / DESIGN_H)
        dw, dh = max(1, int(DESIGN_W * scale)), max(1, int(DESIGN_H * scale))
        scaled = pygame.transform.scale(self.canvas, (dw, dh))
        self.screen.fill(BLACK)
        ox, oy = (sw - dw) // 2, (sh - dh) // 2
        if self.screen_shake > 0:
            mag = int(11 * min(1, self.screen_shake / .48))
            ox += random.randint(-mag, mag); oy += random.randint(-mag, mag)
        self.screen.blit(scaled, (ox, oy))
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.mouse_design = self.design_mouse(pygame.mouse.get_pos())
            self.handle_events()
            self.update(dt)
            self.render()
        self.flush_play_time()
        if self.world and self.lives <= 0 and not self.game_result_recorded:
            self.save.data["stats"]["games_lost"] += 1
            self.game_result_recorded = True
        if self.world and self.save.data.get("campaign", {}).get("active"):
            self.save.data["campaign"] = self.campaign_snapshot(True)
        self.write_settings()
        pygame.quit()

# -----------------------------------------------------------------------------
# Built-in noninteractive checks
# -----------------------------------------------------------------------------
def run_self_tests() -> int:
    import tempfile as _tempfile

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # Save recovery, validation, and atomic round-trip.
    with _tempfile.TemporaryDirectory(prefix="frogvision_test_") as td:
        root = Path(td)
        save_path = root / "presets" / "setsave" / "frogger_save.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("{broken json", encoding="utf-8")
        sm = SaveManager(save_path)
        check(sm.data["settings"]["difficulty"] == "Normal", "corrupt save fallback")
        sm.data["stats"]["high_score"] = 12345
        check(sm.save(), "atomic save write")
        sm2 = SaveManager(save_path)
        check(sm2.data["stats"]["high_score"] == 12345, "save round-trip")
        check(sm2.data["settings"]["background_folder"] == "", "default background folder setting")

        # Every campaign level must generate the full lane set and remain within bounded density.
        for level in range(1, MAX_LEVEL + 1):
            world = LevelWorld(level, "Normal", seed=1000 + level)
            check(len(world.road_lanes) == 5, f"level {level} road lane count")
            check(len(world.river_lanes) == 5, f"level {level} river lane count")
            check(all(len(l.vehicles) >= 2 for l in world.road_lanes), f"level {level} traffic population")
            check(all(len(l.platforms) >= 2 for l in world.river_lanes), f"level {level} platform population")

        # Discrete movement: one command moves exactly one row and clamps arena bounds.
        player = Player()
        check(player.try_hop(0, -1), "player accepts forward hop")
        for _ in range(20):
            player.update(.01)
        check(player.row == START_ROW - 1, "one key press moves one row")
        check(not player.hopping, "hop animation completes")
        player.x = ARENA_X + 4
        check(not player.try_hop(-1, 0), "left boundary clamp")

        # Swept movement catches fast crossings that would miss endpoint-only collision.
        vehicle = Vehicle(ROAD_ROWS[0], 100, "Sports Car", 2000, 1)
        target = pygame.Rect(220, int(vehicle.y), 30, int(vehicle.height))
        vehicle.update_motion(.1, 1.0)
        check(vehicle.swept_rect.colliderect(target), "swept high-speed vehicle collision")

        # Platform carry is frame-rate independent.
        p1 = Platform(RIVER_ROWS[0], 200, "Log", 120, 1)
        p2 = Platform(RIVER_ROWS[0], 200, "Log", 120, 1)
        p1.update_motion(1.0)
        for _ in range(60):
            p2.update_motion(1 / 60)
        check(abs(p1.x - p2.x) < .001, "frame-independent platform movement")

        # Turtle and gator support rules expose safe and unsafe states separately.
        turtle = Platform(RIVER_ROWS[0], 300, "Turtle Group", 90, 1)
        turtle.age = 5.6
        check(not turtle.support_rects(False), "submerged turtle cannot support")
        check(bool(turtle.support_rects(True)), "stable-platform effect restores turtle support")
        gator = Platform(RIVER_ROWS[0], 300, "Cyber-Gator", 90, 1)
        check(bool(gator.support_rects(False)) and bool(gator.unsafe_rects(False)), "gator has body support and unsafe head")

        # Full controller smoke test in a temporary FrameVision root.
        os.environ["FROG_VISION_TEST_ROOT"] = td
        try:
            # No startup art must use the generated fallback.
            game = Game(headless=True)
            check(game.background_name == "Generated fallback", "fallback background without assets")
            check(all(b.action != "continue" for b in game.current_buttons()), "continue hidden without valid progress")
            # A valid startup image must be selected while a corrupt neighbor is skipped safely.
            startup = Path(td) / "presets" / "startup"
            startup.mkdir(parents=True, exist_ok=True)
            test_image = pygame.Surface((320, 200)); test_image.fill((12, 88, 120))
            pygame.image.save(test_image, str(startup / "valid.png"))
            second_image = pygame.Surface((400, 220)); second_image.fill((90, 20, 120))
            pygame.image.save(second_image, str(startup / "second.png"))
            (startup / "corrupt.jpg").write_bytes(b"not an image")
            game_with_art = Game(headless=True)
            check(game_with_art.background_name in {"valid.png", "second.png"}, "startup background loading and corrupt-file skip")
            first_background = game_with_art.background_name
            game = game_with_art
            game.new_game()
            check(game.background_name != first_background, "new level rotates to a different background when possible")
            custom = Path(td) / "custom_backgrounds"
            custom.mkdir(parents=True, exist_ok=True)
            pygame.image.save(test_image, str(custom / "custom.png"))
            game.settings["background_folder"] = str(custom)
            game.background = game.load_background()
            check(game.background_name == "custom.png" and game.background_source == "Custom", "custom background folder loading")
            game.use_default_background_folder()
            check(game.settings["background_folder"] == "" and game.background_source == "Default", "restore default background folder")
            game.state = "title"; game.build_title_buttons()
            check(any(b.action == "continue" for b in game.current_buttons()), "continue appears for valid progress")
            game.state = "level_intro"
            check(game.world is not None and game.state == "level_intro", "game creates level intro")
            game.state = "playing"
            game.player.invulnerable = 0
            initial_lives = game.lives
            game.trigger_death("TIME EXPIRED")
            game.trigger_death("TIME EXPIRED")
            check(game.lives == initial_lives - 1, "death triggers exactly once")
            game.state = "playing"
            game.lives = max(1, game.lives)
            game.filled_slots = {0, 1, 2, 3}
            game.player.row = FINISH_ROW
            game.player.x = game.slot_centers[4] - 18
            game.player.hopping = False
            game.player.invulnerable = 0
            game.handle_finish_landing()
            check(game.state == "level_complete", "fifth slot completes level once")
            old_score = game.score
            game.handle_finish_landing()
            check(game.score == old_score, "level completion cannot duplicate")
            game.level = MAX_LEVEL
            game.state = "level_complete"
            game.advance_level()
            check(game.state == "victory", "level ten advances to victory")
            game.render()
            pygame.quit()
        finally:
            os.environ.pop("FROG_VISION_TEST_ROOT", None)

    if failures:
        print("Frog-Vision self-test FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("Frog-Vision self-test passed: saves, background folders and per-level rotation, 10-level generation, movement, swept collisions, platform rules, death gating, finish slots, victory, and headless rendering.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Frog-Vision neon arcade crossing game")
    parser.add_argument("--self-test", action="store_true", help="run noninteractive smoke and rules tests")
    args = parser.parse_args()
    if args.self_test:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        return run_self_tests()
    game = Game()
    game.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
