#!/usr/bin/env python3
"""Bomber-Vision - a self-contained Bomberman-inspired Pygame game.

Run with:
    python Bomber-vision.py

Controls are also available from the in-game instructions screen.
"""
from __future__ import annotations

import array
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pygame

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

VIRTUAL_W, VIRTUAL_H = 1280, 720
FPS = 60
FIXED_DT = 1.0 / 120.0
GRID_W, GRID_H = 19, 13
TILE = 44
BOARD_X, BOARD_Y = 34, 112
BOARD_W, BOARD_H = GRID_W * TILE, GRID_H * TILE
HUD_X = BOARD_X + BOARD_W + 28
HUD_W = VIRTUAL_W - HUD_X - 28

FLOOR, WALL, BLOCK = 0, 1, 2
DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

COLORS = {
    "bg": (5, 8, 18),
    "panel": (13, 19, 36),
    "panel2": (18, 27, 49),
    "cyan": (60, 238, 255),
    "cyan2": (20, 143, 178),
    "magenta": (255, 63, 190),
    "violet": (142, 85, 255),
    "green": (85, 255, 151),
    "yellow": (255, 221, 88),
    "orange": (255, 132, 56),
    "red": (255, 64, 82),
    "white": (235, 244, 255),
    "muted": (145, 164, 194),
    "floor1": (12, 25, 44),
    "floor2": (15, 32, 54),
    "wall": (23, 50, 76),
    "wall_edge": (53, 191, 220),
    "block": (81, 36, 101),
    "block_edge": (234, 79, 209),
}

DIFFICULTIES = {
    "Easy": {
        "enemy_speed": 0.84,
        "enemy_count": -1,
        "reaction": 1.25,
        "powerup": 1.35,
        "score": 0.8,
        "danger_hints": True,
    },
    "Normal": {
        "enemy_speed": 1.0,
        "enemy_count": 0,
        "reaction": 1.0,
        "powerup": 1.0,
        "score": 1.0,
        "danger_hints": False,
    },
    "Hard": {
        "enemy_speed": 1.18,
        "enemy_count": 2,
        "reaction": 0.72,
        "powerup": 0.72,
        "score": 1.4,
        "danger_hints": False,
    },
}

POWERUP_INFO = {
    "bomb": ("B", COLORS["cyan"], "Bomb capacity +1"),
    "range": ("R", COLORS["orange"], "Blast range +1"),
    "speed": ("S", COLORS["green"], "Movement speed increased"),
    "life": ("+", COLORS["yellow"], "Extra life"),
    "shield": ("O", (120, 195, 255), "Shield online"),
    "fuse": ("F", COLORS["red"], "Shorter bomb fuse"),
    "remote": ("D", COLORS["magenta"], "Remote detonator unlocked"),
    "kick": ("K", COLORS["violet"], "Bomb kick unlocked"),
}

ENEMY_COLORS = {
    "glitch": COLORS["cyan"],
    "hunter": COLORS["red"],
    "panic": COLORS["yellow"],
    "phase": COLORS["violet"],
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def tile_center(tile: Tuple[int, int]) -> pygame.Vector2:
    return pygame.Vector2(tile[0] * TILE + TILE / 2, tile[1] * TILE + TILE / 2)


def world_to_tile(pos: pygame.Vector2) -> Tuple[int, int]:
    return int(pos.x // TILE), int(pos.y // TILE)


def world_rect_to_screen(rect: pygame.Rect) -> pygame.Rect:
    return rect.move(BOARD_X, BOARD_Y)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = COLORS["white"],
    anchor: str = "topleft",
    shadow: bool = False,
) -> pygame.Rect:
    image = font.render(text, True, color)
    rect = image.get_rect()
    setattr(rect, anchor, pos)
    if shadow:
        shadow_img = font.render(text, True, (0, 0, 0))
        shadow_rect = shadow_img.get_rect()
        setattr(shadow_rect, anchor, (pos[0] + 3, pos[1] + 3))
        surface.blit(shadow_img, shadow_rect)
    surface.blit(image, rect)
    return rect


# -----------------------------------------------------------------------------
# Save and audio managers
# -----------------------------------------------------------------------------

class SaveManager:
    DEFAULT = {
        "high_scores": {"Easy": 0, "Normal": 0, "Hard": 0},
        "difficulty": "Normal",
        "volume": 0.35,
        "fullscreen": False,
    }

    def __init__(self) -> None:
        helpers_dir = Path(__file__).resolve().parent
        root_dir = helpers_dir.parent
        self.path = root_dir / "presets" / "setsave" / "bomber_save.json"
        self.data = dict(self.DEFAULT)
        self.data["high_scores"] = dict(self.DEFAULT["high_scores"])
        self.load()

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                scores = raw.get("high_scores", {})
                if isinstance(scores, dict):
                    for key in self.data["high_scores"]:
                        value = scores.get(key)
                        if isinstance(value, int) and value >= 0:
                            self.data["high_scores"][key] = value
                if raw.get("difficulty") in DIFFICULTIES:
                    self.data["difficulty"] = raw["difficulty"]
                volume = raw.get("volume")
                if isinstance(volume, (int, float)):
                    self.data["volume"] = float(clamp(volume, 0.0, 1.0))
                if isinstance(raw.get("fullscreen"), bool):
                    self.data["fullscreen"] = raw["fullscreen"]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Corrupt or inaccessible saves fall back to safe defaults.
            self.data = dict(self.DEFAULT)
            self.data["high_scores"] = dict(self.DEFAULT["high_scores"])

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def high_score(self, difficulty: str) -> int:
        return int(self.data["high_scores"].get(difficulty, 0))

    def submit_score(self, difficulty: str, score: int) -> bool:
        old = self.high_score(difficulty)
        if score > old:
            self.data["high_scores"][difficulty] = int(score)
            self.save()
            return True
        return False


class SoundManager:
    """Creates small synth-like effects in memory; no external files required."""

    def __init__(self, volume: float) -> None:
        self.available = False
        self.muted = False
        self.volume = float(clamp(volume, 0.0, 1.0))
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.available = True
            self._build_sounds()
            self.set_volume(self.volume)
        except pygame.error:
            self.available = False

    @staticmethod
    def _wave(
        frequency: float,
        duration: float,
        volume: float = 0.35,
        kind: str = "sine",
        slide: float = 0.0,
        attack: float = 0.01,
    ) -> pygame.mixer.Sound:
        sample_rate = 44100
        count = max(1, int(duration * sample_rate))
        buf = array.array("h")
        phase = 0.0
        for i in range(count):
            t = i / sample_rate
            f = max(30.0, frequency + slide * (t / max(duration, 0.001)))
            phase += 2 * math.pi * f / sample_rate
            if kind == "square":
                raw = 1.0 if math.sin(phase) >= 0 else -1.0
            elif kind == "noise":
                raw = random.uniform(-1.0, 1.0)
            else:
                raw = math.sin(phase)
            fade_in = min(1.0, t / max(attack, 0.001))
            fade_out = max(0.0, 1.0 - (t / duration)) ** 1.7
            sample = int(32767 * volume * raw * fade_in * fade_out)
            channels = pygame.mixer.get_init()[2] if pygame.mixer.get_init() else 1
            for _ in range(max(1, channels)):
                buf.append(sample)
        return pygame.mixer.Sound(buffer=buf.tobytes())

    def _build_sounds(self) -> None:
        specs = {
            "menu": (620, 0.08, 0.22, "sine", 160),
            "place": (185, 0.11, 0.32, "square", -45),
            "fuse": (880, 0.05, 0.15, "square", -120),
            "explode": (90, 0.30, 0.38, "noise", -30),
            "block": (240, 0.10, 0.24, "noise", -120),
            "enemy": (430, 0.18, 0.28, "square", -320),
            "pickup": (620, 0.18, 0.24, "sine", 520),
            "hit": (150, 0.32, 0.32, "square", -100),
            "level": (520, 0.55, 0.22, "sine", 700),
            "gameover": (330, 0.75, 0.25, "square", -260),
        }
        for name, spec in specs.items():
            self.sounds[name] = self._wave(*spec)

    def play(self, name: str) -> None:
        if self.available and not self.muted:
            sound = self.sounds.get(name)
            if sound:
                sound.play()

    def set_volume(self, value: float) -> None:
        self.volume = float(clamp(value, 0.0, 1.0))
        if self.available:
            for sound in self.sounds.values():
                sound.set_volume(self.volume)

    def toggle_mute(self) -> None:
        self.muted = not self.muted


# -----------------------------------------------------------------------------
# Visual helpers
# -----------------------------------------------------------------------------

@dataclass
class Particle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    color: Tuple[int, int, int]
    life: float
    max_life: float
    size: float
    gravity: float = 0.0

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.vel.y += self.gravity * dt
        self.pos += self.vel * dt
        return self.life > 0

    def draw(self, surface: pygame.Surface) -> None:
        alpha = int(255 * clamp(self.life / self.max_life, 0.0, 1.0))
        radius = max(1, int(self.size * (0.45 + 0.55 * self.life / self.max_life)))
        temp = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(temp, (*self.color, alpha), (radius * 2, radius * 2), radius)
        surface.blit(temp, (int(self.pos.x) - radius * 2, int(self.pos.y) - radius * 2))


@dataclass
class FloatingText:
    text: str
    pos: pygame.Vector2
    color: Tuple[int, int, int]
    life: float = 1.3

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.pos.y -= 28 * dt
        return self.life > 0

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        alpha = int(255 * clamp(self.life / 1.3, 0.0, 1.0))
        img = font.render(self.text, True, self.color)
        img.set_alpha(alpha)
        rect = img.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(img, rect)


# -----------------------------------------------------------------------------
# Arena and entities
# -----------------------------------------------------------------------------

class Arena:
    def __init__(self, level: int, rng: random.Random) -> None:
        self.level = level
        self.rng = rng
        self.grid: List[List[int]] = [[FLOOR for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.destroyed_blocks: Set[Tuple[int, int]] = set()
        self.exit_tile: Tuple[int, int] = (GRID_W - 2, GRID_H - 2)
        self.exit_revealed = False
        self.exit_active = False
        self.unstable_tiles: Set[Tuple[int, int]] = set()
        self._generate()

    def _generate(self) -> None:
        for y in range(GRID_H):
            for x in range(GRID_W):
                if x == 0 or y == 0 or x == GRID_W - 1 or y == GRID_H - 1:
                    self.grid[y][x] = WALL
                elif x % 2 == 0 and y % 2 == 0:
                    self.grid[y][x] = WALL

        safe = {(1, 1), (2, 1), (1, 2), (GRID_W - 2, GRID_H - 2), (GRID_W - 3, GRID_H - 2), (GRID_W - 2, GRID_H - 3)}
        density = clamp(0.35 + self.level * 0.018, 0.35, 0.53)
        candidates: List[Tuple[int, int]] = []
        for y in range(1, GRID_H - 1):
            for x in range(1, GRID_W - 1):
                if self.grid[y][x] == FLOOR and (x, y) not in safe:
                    candidates.append((x, y))
                    if self.rng.random() < density:
                        self.grid[y][x] = BLOCK

        # Exit is always under a block, and the surrounding end area is reachable
        # after destruction because permanent walls never isolate floor components.
        far_candidates = [p for p in candidates if abs(p[0] - 1) + abs(p[1] - 1) > 12]
        self.exit_tile = self.rng.choice(far_candidates or candidates)
        ex, ey = self.exit_tile
        self.grid[ey][ex] = BLOCK

        if self.level == 9:
            open_tiles = [p for p in self.floor_tiles() if p not in safe and p != self.exit_tile]
            self.rng.shuffle(open_tiles)
            self.unstable_tiles = set(open_tiles[:14])

    def inside(self, tile: Tuple[int, int]) -> bool:
        x, y = tile
        return 0 <= x < GRID_W and 0 <= y < GRID_H

    def tile_value(self, tile: Tuple[int, int]) -> int:
        x, y = tile
        if not self.inside(tile):
            return WALL
        return self.grid[y][x]

    def is_solid(self, tile: Tuple[int, int], phase_blocks: bool = False) -> bool:
        value = self.tile_value(tile)
        if value == WALL:
            return True
        if value == BLOCK and not phase_blocks:
            return True
        return False

    def floor_tiles(self) -> List[Tuple[int, int]]:
        return [(x, y) for y in range(1, GRID_H - 1) for x in range(1, GRID_W - 1) if self.grid[y][x] == FLOOR]

    def destroy_block(self, tile: Tuple[int, int]) -> bool:
        if tile in self.destroyed_blocks:
            return False
        x, y = tile
        if self.inside(tile) and self.grid[y][x] == BLOCK:
            self.grid[y][x] = FLOOR
            self.destroyed_blocks.add(tile)
            if tile == self.exit_tile:
                self.exit_revealed = True
            return True
        return False

    def nearest_safe_tile(self, origin: Tuple[int, int], forbidden: Set[Tuple[int, int]]) -> Tuple[int, int]:
        queue = deque([origin])
        seen = {origin}
        while queue:
            tile = queue.popleft()
            if self.inside(tile) and self.tile_value(tile) == FLOOR and tile not in forbidden:
                return tile
            for dx, dy in DIRS:
                nxt = (tile[0] + dx, tile[1] + dy)
                if nxt not in seen and self.inside(nxt):
                    seen.add(nxt)
                    queue.append(nxt)
        return (1, 1)

    def draw(self, surface: pygame.Surface, time_s: float) -> None:
        # Draw the arena on an alpha layer so FrameVision's selected startup
        # image remains visible through the floor and scenery. Walls and blocks
        # stay opaque enough for reliable collision reading.
        layer = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                checker = (12, 25, 44, 82) if (x + y) % 2 == 0 else (15, 32, 54, 82)
                pygame.draw.rect(layer, checker, rect)
                pygame.draw.rect(layer, (30, 92, 119, 112), rect, 1)
                value = self.grid[y][x]
                if value == WALL:
                    pygame.draw.rect(layer, (*COLORS["wall"], 210), rect.inflate(-4, -4), border_radius=6)
                    pygame.draw.rect(layer, (*COLORS["wall_edge"], 245), rect.inflate(-7, -7), 2, border_radius=6)
                    inner = rect.inflate(-16, -16)
                    pygame.draw.rect(layer, (10, 26, 43, 180), inner, border_radius=4)
                elif value == BLOCK:
                    pulse = int(20 + 20 * (0.5 + 0.5 * math.sin(time_s * 2.0 + x + y)))
                    base = (COLORS["block"][0] + pulse, COLORS["block"][1], COLORS["block"][2] + pulse // 2, 215)
                    pygame.draw.rect(layer, base, rect.inflate(-5, -5), border_radius=4)
                    pygame.draw.rect(layer, (*COLORS["block_edge"], 245), rect.inflate(-8, -8), 2, border_radius=4)
                    pygame.draw.line(layer, (150, 65, 160, 225), rect.topleft + pygame.Vector2(10, 10), rect.bottomright - pygame.Vector2(10, 10), 2)
                    pygame.draw.line(layer, (150, 65, 160, 225), (rect.right - 10, rect.top + 10), (rect.left + 10, rect.bottom - 10), 2)
                elif (x, y) in self.unstable_tiles:
                    a = 0.5 + 0.5 * math.sin(time_s * 3.2 + x * 0.7)
                    color = (int(60 + 130 * a), int(30 + 40 * a), int(45 + 25 * a), 230)
                    pygame.draw.rect(layer, color, rect.inflate(-8, -8), 2, border_radius=5)

        if self.exit_revealed:
            x, y = self.exit_tile
            rect = pygame.Rect(x * TILE + 7, y * TILE + 7, TILE - 14, TILE - 14)
            color = COLORS["green"] if self.exit_active else COLORS["red"]
            pygame.draw.rect(layer, (12, 17, 30, 225), rect, border_radius=7)
            pygame.draw.rect(layer, (*color, 255), rect, 3, border_radius=7)
            pygame.draw.circle(layer, (*color, 255), rect.center, 6)

        surface.blit(layer, (BOARD_X, BOARD_Y))


class Actor:
    _next_id = 1

    def __init__(self, tile: Tuple[int, int], radius: float) -> None:
        self.id = Actor._next_id
        Actor._next_id += 1
        self.pos = tile_center(tile)
        self.radius = radius
        self.alive = True

    @property
    def tile(self) -> Tuple[int, int]:
        return world_to_tile(self.pos)

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.pos.x - self.radius), int(self.pos.y - self.radius), int(self.radius * 2), int(self.radius * 2))


class Player(Actor):
    def __init__(self, tile: Tuple[int, int]) -> None:
        super().__init__(tile, 14)
        self.spawn_tile = tile
        self.lives = 3
        self.bomb_capacity = 1
        self.blast_range = 2
        self.speed_level = 0
        self.fuse_level = 0
        self.has_remote = False
        self.has_kick = False
        self.shield_time = 0.0
        self.invulnerable = 0.0
        self.can_pass_bombs: Set[int] = set()
        self.facing = pygame.Vector2(0, 1)

    @property
    def move_speed(self) -> float:
        return 185.0 + self.speed_level * 22.0

    @property
    def fuse_time(self) -> float:
        return max(1.25, 2.55 - self.fuse_level * 0.23)

    def update_timers(self, dt: float) -> None:
        self.invulnerable = max(0.0, self.invulnerable - dt)
        self.shield_time = max(0.0, self.shield_time - dt)

    def draw(self, surface: pygame.Surface, time_s: float) -> None:
        if self.invulnerable > 0 and int(time_s * 14) % 2 == 0:
            return
        center = (BOARD_X + int(self.pos.x), BOARD_Y + int(self.pos.y))
        if self.shield_time > 0:
            radius = int(21 + 2 * math.sin(time_s * 6))
            pygame.draw.circle(surface, (80, 175, 255), center, radius, 2)
        pygame.draw.circle(surface, (8, 20, 35), center, 17)
        pygame.draw.circle(surface, COLORS["cyan"], center, 14)
        pygame.draw.circle(surface, (17, 59, 84), center, 9)
        eye_offset = self.facing * 5
        eye = (int(center[0] + eye_offset.x), int(center[1] + eye_offset.y))
        pygame.draw.circle(surface, COLORS["white"], eye, 4)
        pygame.draw.circle(surface, COLORS["magenta"], eye, 2)


@dataclass
class Bomb:
    id: int
    owner_id: int
    tile: Tuple[int, int]
    timer: float
    blast_range: int
    serial: int
    move_dir: Optional[Tuple[int, int]] = None
    move_progress: float = 0.0
    fuse_pinged: bool = False

    def world_pos(self) -> pygame.Vector2:
        if self.move_dir:
            start = tile_center(self.tile)
            end = tile_center((self.tile[0] + self.move_dir[0], self.tile[1] + self.move_dir[1]))
            return start.lerp(end, self.move_progress)
        return tile_center(self.tile)

    def occupied_tiles(self) -> Set[Tuple[int, int]]:
        occupied = {self.tile}
        if self.move_dir and self.move_progress > 0.12:
            occupied.add((self.tile[0] + self.move_dir[0], self.tile[1] + self.move_dir[1]))
        return occupied

    def draw(self, surface: pygame.Surface, time_s: float) -> None:
        p = self.world_pos()
        center = (BOARD_X + int(p.x), BOARD_Y + int(p.y))
        pulse = 1.0 + 0.12 * math.sin(time_s * 9 + self.id)
        radius = int(13 * pulse)
        pygame.draw.circle(surface, (8, 9, 15), center, radius + 3)
        pygame.draw.circle(surface, (50, 56, 77), center, radius)
        pygame.draw.circle(surface, COLORS["magenta"], (center[0] - 4, center[1] - 4), 4)
        fuse_end = (center[0] + 11, center[1] - 14)
        pygame.draw.line(surface, COLORS["yellow"], (center[0] + 5, center[1] - 9), fuse_end, 3)
        spark = int(3 + 2 * math.sin(time_s * 18 + self.id))
        pygame.draw.circle(surface, COLORS["orange"], fuse_end, spark)


@dataclass
class Explosion:
    id: int
    tiles: Set[Tuple[int, int]]
    life: float = 0.52
    max_life: float = 0.52
    hit_actors: Set[int] = field(default_factory=set)

    def update(self, dt: float) -> bool:
        self.life -= dt
        return self.life > 0

    def draw(self, surface: pygame.Surface, time_s: float) -> None:
        ratio = clamp(self.life / self.max_life, 0.0, 1.0)
        for x, y in self.tiles:
            rect = pygame.Rect(BOARD_X + x * TILE + 4, BOARD_Y + y * TILE + 4, TILE - 8, TILE - 8)
            pygame.draw.rect(surface, COLORS["red"], rect, border_radius=9)
            inner = rect.inflate(-10, -10)
            pygame.draw.rect(surface, COLORS["yellow"], inner, border_radius=7)
            core = inner.inflate(-10, -10)
            pygame.draw.rect(surface, COLORS["white"], core, border_radius=5)
            if ratio < 0.35:
                veil = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                veil.fill((255, 60, 60, int(180 * (1 - ratio / 0.35))))
                surface.blit(veil, rect)


@dataclass
class PowerUp:
    kind: str
    tile: Tuple[int, int]
    bob: float = 0.0

    def update(self, dt: float) -> None:
        self.bob += dt

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        symbol, color, _ = POWERUP_INFO[self.kind]
        center = (BOARD_X + int((self.tile[0] + 0.5) * TILE), BOARD_Y + int((self.tile[1] + 0.5) * TILE + math.sin(self.bob * 4) * 3))
        pygame.draw.circle(surface, (8, 14, 25), center, 16)
        pygame.draw.circle(surface, color, center, 14, 3)
        draw_text(surface, font, symbol, center, color, "center")

class Enemy(Actor):
    kind = "glitch"
    base_speed = 100.0
    score_value = 300

    def __init__(self, tile: Tuple[int, int], difficulty: str, level: int, rng: random.Random) -> None:
        super().__init__(tile, 13)
        self.rng = rng
        self.difficulty = difficulty
        self.level = level
        self.direction = pygame.Vector2(0, 0)
        self.target_tile = tile
        self.decision_timer = self.rng.uniform(0.05, 0.4)
        self.stuck_timer = 0.0
        self.phase_active = False
        self.phase_timer = 0.0
        self.phase_cooldown = self.rng.uniform(1.0, 3.0)

    @property
    def speed(self) -> float:
        diff = DIFFICULTIES[self.difficulty]["enemy_speed"]
        return self.base_speed * diff * (1.0 + min(self.level - 1, 9) * 0.025)

    def can_phase_blocks(self) -> bool:
        return False

    def choose_direction(self, session: "GameSession") -> Tuple[int, int]:
        options = session.valid_enemy_moves(self.tile, self.can_phase_blocks(), self.id)
        if not options:
            return (0, 0)
        current = (int(self.direction.x), int(self.direction.y))
        reverse = (-current[0], -current[1])
        non_reverse = [d for d in options if d != reverse]
        return self.rng.choice(non_reverse or options)

    def update(self, dt: float, session: "GameSession") -> None:
        if not self.alive:
            return
        self.decision_timer -= dt
        self._update_special(dt, session)

        center = tile_center(self.tile)
        at_center = self.pos.distance_to(center) < 2.0
        if at_center:
            self.pos = center
            if self.decision_timer <= 0 or self.direction.length_squared() == 0:
                dx, dy = self.choose_direction(session)
                self.direction.update(dx, dy)
                reaction = DIFFICULTIES[self.difficulty]["reaction"]
                self.decision_timer = max(0.08, self.rng.uniform(0.22, 0.55) * reaction)

            if self.direction.length_squared() > 0:
                next_tile = (self.tile[0] + int(self.direction.x), self.tile[1] + int(self.direction.y))
                if session.enemy_tile_blocked(next_tile, self.can_phase_blocks(), self.id):
                    self.direction.update(0, 0)
                    self.decision_timer = 0
                    return

        move = self.direction * self.speed * dt
        old = self.pos.copy()
        self.pos += move
        # Snap if we crossed the target center.
        if self.direction.length_squared() > 0:
            old_tile = world_to_tile(old)
            new_tile = world_to_tile(self.pos)
            if old_tile != new_tile:
                next_center = tile_center(new_tile)
                if (self.pos - next_center).dot(self.direction) >= 0:
                    self.pos = next_center
                    self.decision_timer = min(self.decision_timer, 0.0)
        if self.pos.distance_to(old) < 0.01:
            self.stuck_timer += dt
            if self.stuck_timer > 0.3:
                self.direction.update(0, 0)
                self.decision_timer = 0
        else:
            self.stuck_timer = 0.0

    def _update_special(self, dt: float, session: "GameSession") -> None:
        return

    def draw(self, surface: pygame.Surface, time_s: float) -> None:
        center = (BOARD_X + int(self.pos.x), BOARD_Y + int(self.pos.y))
        color = ENEMY_COLORS[self.kind]
        pygame.draw.circle(surface, (5, 10, 18), center, 17)
        if self.kind == "glitch":
            pygame.draw.rect(surface, color, (center[0] - 12, center[1] - 10, 24, 20), border_radius=5)
            pygame.draw.line(surface, COLORS["white"], (center[0] - 7, center[1]), (center[0] + 7, center[1]), 2)
        elif self.kind == "hunter":
            points = [(center[0], center[1] - 15), (center[0] + 14, center[1] + 11), (center[0] - 14, center[1] + 11)]
            pygame.draw.polygon(surface, color, points)
            pygame.draw.circle(surface, COLORS["white"], center, 4)
        elif self.kind == "panic":
            pygame.draw.circle(surface, color, center, 13)
            pygame.draw.circle(surface, (15, 20, 25), (center[0] - 5, center[1] - 2), 3)
            pygame.draw.circle(surface, (15, 20, 25), (center[0] + 5, center[1] - 2), 3)
            pygame.draw.arc(surface, (15, 20, 25), (center[0] - 7, center[1] + 1, 14, 9), math.pi, math.pi * 2, 2)
        else:
            alpha_color = (180, 135, 255) if self.phase_active else color
            points = []
            for i in range(6):
                ang = math.pi / 3 * i + time_s * 0.5
                points.append((center[0] + math.cos(ang) * 14, center[1] + math.sin(ang) * 14))
            pygame.draw.polygon(surface, alpha_color, points, 3 if self.phase_active else 0)
            pygame.draw.circle(surface, COLORS["white"], center, 4)


class GlitchDrone(Enemy):
    kind = "glitch"
    base_speed = 90.0
    score_value = 250


class HunterBot(Enemy):
    kind = "hunter"
    base_speed = 108.0
    score_value = 450

    def choose_direction(self, session: "GameSession") -> Tuple[int, int]:
        path = session.find_path(self.tile, session.player.tile, avoid_danger=True, phase_blocks=False)
        if len(path) >= 2:
            nxt = path[1]
            return nxt[0] - self.tile[0], nxt[1] - self.tile[1]
        return super().choose_direction(session)


class PanicUnit(Enemy):
    kind = "panic"
    base_speed = 116.0
    score_value = 400

    def choose_direction(self, session: "GameSession") -> Tuple[int, int]:
        options = session.valid_enemy_moves(self.tile, False, self.id)
        if not options:
            return (0, 0)
        danger = session.danger_map
        threatened = danger.get(self.tile, 99.0) < 1.5
        if threatened:
            scored: List[Tuple[float, Tuple[int, int]]] = []
            for d in options:
                tile = (self.tile[0] + d[0], self.tile[1] + d[1])
                score = danger.get(tile, 99.0)
                score += 0.08 * abs(tile[0] - session.player.tile[0])
                scored.append((score + self.rng.random() * 0.05, d))
            return max(scored, key=lambda item: item[0])[1]
        return super().choose_direction(session)


class PhaseStalker(Enemy):
    kind = "phase"
    base_speed = 102.0
    score_value = 700

    def can_phase_blocks(self) -> bool:
        return self.phase_active

    def _update_special(self, dt: float, session: "GameSession") -> None:
        self.phase_cooldown -= dt
        if self.phase_active:
            self.phase_timer -= dt
            if self.phase_timer <= 0:
                if session.arena.tile_value(self.tile) == FLOOR:
                    self.phase_active = False
                    self.phase_cooldown = self.rng.uniform(2.5, 4.5)
                else:
                    self.phase_timer = 0.35
        elif self.phase_cooldown <= 0:
            self.phase_active = True
            self.phase_timer = self.rng.uniform(1.4, 2.2)
            self.decision_timer = 0

    def choose_direction(self, session: "GameSession") -> Tuple[int, int]:
        path = session.find_path(self.tile, session.player.tile, avoid_danger=True, phase_blocks=self.phase_active)
        if len(path) >= 2:
            nxt = path[1]
            return nxt[0] - self.tile[0], nxt[1] - self.tile[1]
        return super().choose_direction(session)


class GameSession:
    """Owns every run-specific object so restart and level transitions are clean."""

    def __init__(self, game: "Game", difficulty: str) -> None:
        self.game = game
        self.difficulty = difficulty
        self.rng = random.Random()
        self.level = 1
        self.score = 0
        self.player = Player((1, 1))
        self.arena = Arena(self.level, self.rng)
        self.bombs: List[Bomb] = []
        self.explosions: List[Explosion] = []
        self.enemies: List[Enemy] = []
        self.powerups: List[PowerUp] = []
        self.particles: List[Particle] = []
        self.floaters: List[FloatingText] = []
        self.notifications: deque[Tuple[str, float, Tuple[int, int, int]]] = deque()
        self.bomb_counter = 1
        self.explosion_counter = 1
        self.bomb_serial = 1
        self.elapsed = 0.0
        self.level_elapsed = 0.0
        self.combo_count = 0
        self.combo_multiplier = 1
        self.combo_timer = 0.0
        self.next_life_score = 20000
        self.level_damage_taken = False
        self.pending_complete = False
        self.complete_timer = 0.0
        self.danger_map: Dict[Tuple[int, int], float] = {}
        self.danger_refresh = 0.0
        self.challenge = ""
        self.challenge_timer = 0.0
        self.reinforcement_done = False
        self.reactor_cycle = 8.0
        self.reactor_warning = 0.0
        self.reactor_active_tiles: Set[Tuple[int, int]] = set()
        self.start_level(1, preserve_player=False)

    def challenge_for_level(self, level: int) -> str:
        return {
            3: "OVERCLOCKED FUSES: every bomb burns faster",
            6: "REINFORCEMENT SIGNAL: more enemies arrive mid-level",
            8: "GRID BLACKOUT: visibility pulses across the arena",
            9: "UNSTABLE REACTOR: marked floor tiles periodically ignite",
        }.get(level, "")

    def start_level(self, level: int, preserve_player: bool = True) -> None:
        self.level = level
        if not preserve_player:
            self.player = Player((1, 1))
        self.player.pos = tile_center((1, 1))
        self.player.spawn_tile = (1, 1)
        self.player.invulnerable = 1.2
        self.player.can_pass_bombs.clear()
        self.arena = Arena(level, self.rng)
        self.bombs = []
        self.explosions = []
        self.powerups = []
        self.particles = []
        self.floaters = []
        self.enemies = []
        self.pending_complete = False
        self.complete_timer = 0.0
        self.level_elapsed = 0.0
        self.level_damage_taken = False
        self.challenge = self.challenge_for_level(level)
        self.challenge_timer = 0.0
        self.reinforcement_done = False
        self.reactor_cycle = 7.5
        self.reactor_warning = 0.0
        self.reactor_active_tiles.clear()
        self._spawn_enemies()
        self.arena.exit_active = False
        self.danger_map = {}
        self.danger_refresh = 0.0
        self.notifications.clear()
        self.notify(f"LEVEL {level}", 2.0, COLORS["cyan"])
        if self.challenge:
            self.notify(self.challenge, 3.4, COLORS["yellow"])

    def _spawn_enemies(self) -> None:
        base = 2 + self.level // 2 + int(DIFFICULTIES[self.difficulty]["enemy_count"])
        count = int(clamp(base, 2, 10))
        candidates = [
            tile for tile in self.arena.floor_tiles()
            if abs(tile[0] - 1) + abs(tile[1] - 1) >= 8 and tile != self.arena.exit_tile
        ]
        self.rng.shuffle(candidates)
        classes: List[type[Enemy]] = [GlitchDrone]
        if self.level >= 2:
            classes.append(HunterBot)
        if self.level >= 4:
            classes.append(PanicUnit)
        if self.level >= 7:
            classes.append(PhaseStalker)
        for i in range(min(count, len(candidates))):
            cls = classes[min(len(classes) - 1, (i + self.level // 2) % len(classes))]
            self.enemies.append(cls(candidates[i], self.difficulty, self.level, self.rng))

    def notify(self, text: str, duration: float = 1.8, color: Tuple[int, int, int] = COLORS["white"]) -> None:
        self.notifications.append((text, duration, color))

    def add_score(self, base: int, combo_action: bool = False, at: Optional[pygame.Vector2] = None) -> None:
        if combo_action:
            if self.combo_timer > 0:
                self.combo_count += 1
            else:
                self.combo_count = 1
            self.combo_timer = 2.5
            self.combo_multiplier = min(8, 1 + self.combo_count // 3)
        value = int(base * self.combo_multiplier * DIFFICULTIES[self.difficulty]["score"])
        self.score += value
        if at is not None:
            self.floaters.append(FloatingText(f"+{value}", at + pygame.Vector2(BOARD_X, BOARD_Y), COLORS["yellow"]))
        while self.score >= self.next_life_score:
            self.player.lives += 1
            self.next_life_score += 20000
            self.notify("EXTRA LIFE: score milestone", 2.2, COLORS["yellow"])
            self.game.sound.play("pickup")

    def update(self, dt: float, movement: pygame.Vector2) -> None:
        self.elapsed += dt
        self.level_elapsed += dt
        self.player.update_timers(dt)
        self._update_notifications(dt)
        self.combo_timer = max(0.0, self.combo_timer - dt)
        if self.combo_timer <= 0:
            self.combo_count = 0
            self.combo_multiplier = 1

        self._move_player(dt, movement)
        self._update_bombs(dt)
        self._update_explosions(dt)
        if self.game.state != "playing":
            return
        self._update_challenge(dt)

        self.danger_refresh -= dt
        if self.danger_refresh <= 0:
            self.danger_map = self.build_danger_map()
            self.danger_refresh = 0.18

        for enemy in list(self.enemies):
            enemy.update(dt, self)
        self._resolve_enemy_contacts()
        if self.game.state != "playing":
            return
        self._collect_powerups()

        for powerup in self.powerups:
            powerup.update(dt)
        self.particles = [p for p in self.particles if p.update(dt)]
        self.floaters = [f for f in self.floaters if f.update(dt)]

        if not self.enemies:
            self.arena.exit_active = self.arena.exit_revealed
        if self.arena.exit_active and self.player.tile == self.arena.exit_tile and not self.pending_complete:
            self.pending_complete = True
            self.complete_timer = 0.8
        if self.pending_complete:
            self.complete_timer -= dt
            if self.complete_timer <= 0:
                self.finish_level()

    def _update_notifications(self, dt: float) -> None:
        if not self.notifications:
            return
        text, remaining, color = self.notifications[0]
        remaining -= dt
        if remaining <= 0:
            self.notifications.popleft()
        else:
            self.notifications[0] = (text, remaining, color)

    def _player_bombs(self) -> List[Bomb]:
        return [bomb for bomb in self.bombs if bomb.owner_id == self.player.id]

    def place_or_remote(self) -> None:
        tile = self.player.tile
        active = self._player_bombs()
        # Remote detonation takes priority once capacity is full, including
        # while the player is still stepping off the newest bomb tile.
        if len(active) >= self.player.bomb_capacity:
            if self.player.has_remote and active:
                oldest = min(active, key=lambda bomb: bomb.serial)
                oldest.timer = 0.0
            return
        if any(tile in bomb.occupied_tiles() for bomb in self.bombs):
            return
        if self.arena.tile_value(tile) != FLOOR:
            return
        fuse = self.player.fuse_time
        if self.level == 3:
            fuse *= 0.72
        bomb = Bomb(
            id=self.bomb_counter,
            owner_id=self.player.id,
            tile=tile,
            timer=fuse,
            blast_range=self.player.blast_range,
            serial=self.bomb_serial,
        )
        self.bomb_counter += 1
        self.bomb_serial += 1
        self.bombs.append(bomb)
        self.player.can_pass_bombs.add(bomb.id)
        self.game.sound.play("place")

    def bomb_at(self, tile: Tuple[int, int], ignore_id: Optional[int] = None) -> Optional[Bomb]:
        for bomb in self.bombs:
            if bomb.id == ignore_id:
                continue
            if tile in bomb.occupied_tiles():
                return bomb
        return None

    def _rect_collides_world(self, rect: pygame.Rect, ignore_bombs: Set[int]) -> Optional[Bomb]:
        left = max(0, rect.left // TILE)
        right = min(GRID_W - 1, rect.right // TILE)
        top = max(0, rect.top // TILE)
        bottom = min(GRID_H - 1, rect.bottom // TILE)
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                if self.arena.is_solid((x, y)):
                    tile_rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                    if rect.colliderect(tile_rect):
                        return Bomb(-1, -1, (x, y), 0, 0, 0)
        for bomb in self.bombs:
            if bomb.id in ignore_bombs:
                continue
            p = bomb.world_pos()
            bomb_rect = pygame.Rect(int(p.x - 15), int(p.y - 15), 30, 30)
            if rect.colliderect(bomb_rect):
                return bomb
        return None

    def _move_player(self, dt: float, movement: pygame.Vector2) -> None:
        if movement.length_squared() > 1:
            movement = movement.normalize()
        if movement.length_squared() > 0:
            self.player.facing = movement.normalize()
        delta = movement * self.player.move_speed * dt
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-5:
                continue
            trial = self.player.pos.copy()
            trial[axis] += delta[axis]
            rect = pygame.Rect(int(trial.x - self.player.radius), int(trial.y - self.player.radius), int(self.player.radius * 2), int(self.player.radius * 2))
            collision = self._rect_collides_world(rect, self.player.can_pass_bombs)
            if collision and collision.id > 0 and self.player.has_kick:
                kick_dir = (int(math.copysign(1, delta.x)), 0) if axis == 0 else (0, int(math.copysign(1, delta.y)))
                if self.kick_bomb(collision, kick_dir):
                    collision = None
            if collision is None:
                self.player.pos = trial

        for bomb_id in list(self.player.can_pass_bombs):
            bomb = next((b for b in self.bombs if b.id == bomb_id), None)
            if bomb is None:
                self.player.can_pass_bombs.discard(bomb_id)
                continue
            # Keep pass-through permission until the player's collision circle
            # has fully cleared the newly placed bomb, not merely until the
            # grid tile changes. Removing it at the tile boundary can trap the
            # player while the two shapes still overlap.
            if self.player.pos.distance_to(bomb.world_pos()) > self.player.radius + 17:
                self.player.can_pass_bombs.discard(bomb_id)

    def kick_bomb(self, bomb: Bomb, direction: Tuple[int, int]) -> bool:
        if bomb.move_dir is not None or direction == (0, 0):
            return False
        target = (bomb.tile[0] + direction[0], bomb.tile[1] + direction[1])
        if self.bomb_move_blocked(target, bomb.id):
            return False
        bomb.move_dir = direction
        bomb.move_progress = 0.0
        return True

    def bomb_move_blocked(self, tile: Tuple[int, int], bomb_id: int) -> bool:
        if self.arena.is_solid(tile):
            return True
        if self.bomb_at(tile, ignore_id=bomb_id):
            return True
        if tile == self.player.tile:
            return True
        return False

    def _update_bombs(self, dt: float) -> None:
        initial: List[Bomb] = []
        for bomb in list(self.bombs):
            bomb.timer -= dt
            if bomb.timer < 0.55 and not bomb.fuse_pinged:
                bomb.fuse_pinged = True
                self.game.sound.play("fuse")
            if bomb.move_dir:
                target = (bomb.tile[0] + bomb.move_dir[0], bomb.tile[1] + bomb.move_dir[1])
                # A kicked bomb stops before a player, wall, block, or bomb that
                # entered its target after the kick began.
                if self.bomb_move_blocked(target, bomb.id):
                    bomb.move_dir = None
                    bomb.move_progress = 0.0
                else:
                    bomb.move_progress += (320.0 / TILE) * dt
                    while bomb.move_progress >= 1.0 and bomb.move_dir:
                        bomb.move_progress -= 1.0
                        bomb.tile = (bomb.tile[0] + bomb.move_dir[0], bomb.tile[1] + bomb.move_dir[1])
                        nxt = (bomb.tile[0] + bomb.move_dir[0], bomb.tile[1] + bomb.move_dir[1])
                        if self.bomb_move_blocked(nxt, bomb.id):
                            bomb.move_dir = None
                            bomb.move_progress = 0.0
            if bomb.timer <= 0:
                initial.append(bomb)
        if initial:
            self._detonate_chain(initial)

    def _blast_tiles(self, bomb: Bomb, grid_snapshot: Optional[List[List[int]]] = None) -> Set[Tuple[int, int]]:
        origin = bomb.tile
        if bomb.move_dir and bomb.move_progress >= 0.5:
            origin = (origin[0] + bomb.move_dir[0], origin[1] + bomb.move_dir[1])
        tiles = {origin}
        for dx, dy in DIRS:
            for step in range(1, bomb.blast_range + 1):
                tile = (origin[0] + dx * step, origin[1] + dy * step)
                if grid_snapshot is None:
                    value = self.arena.tile_value(tile)
                else:
                    x, y = tile
                    value = grid_snapshot[y][x] if 0 <= x < GRID_W and 0 <= y < GRID_H else WALL
                if value == WALL:
                    break
                tiles.add(tile)
                if value == BLOCK:
                    break
        return tiles

    def _detonate_chain(self, initial: Sequence[Bomb]) -> None:
        queue = deque(initial)
        # Every bomb detonated in this fixed step sees the same destructible-wall
        # layout. This prevents a later arm in the same chain from incorrectly
        # travelling through a block destroyed a few microseconds earlier.
        grid_snapshot = [row[:] for row in self.arena.grid]
        processed: Set[int] = set()
        chain_size = 0
        while queue:
            bomb = queue.popleft()
            if bomb.id in processed or bomb not in self.bombs:
                continue
            processed.add(bomb.id)
            chain_size += 1
            tiles = self._blast_tiles(bomb, grid_snapshot)
            self.bombs.remove(bomb)
            self.player.can_pass_bombs.discard(bomb.id)

            destroyed_now: List[Tuple[int, int]] = []
            for tile in tiles:
                if self.arena.destroy_block(tile):
                    destroyed_now.append(tile)
                    self.add_score(50, combo_action=True, at=tile_center(tile))
                    self.game.sound.play("block")
                    self._spawn_debris(tile, COLORS["magenta"], 8)
                    self._maybe_spawn_powerup(tile)
            for other in list(self.bombs):
                if other.tile in tiles or any(t in tiles for t in other.occupied_tiles()):
                    other.timer = 0.0
                    queue.append(other)
            explosion = Explosion(self.explosion_counter, tiles)
            self.explosion_counter += 1
            self.explosions.append(explosion)
            self._spawn_explosion_particles(tiles)
            self.game.sound.play("explode")
            self.game.shake = max(self.game.shake, 7.0)

        if chain_size > 1:
            bonus = 100 * chain_size * chain_size
            self.add_score(bonus, combo_action=True, at=self.player.pos.copy())
            self.notify(f"CHAIN REACTION x{chain_size}", 1.8, COLORS["orange"])

    def _update_explosions(self, dt: float) -> None:
        for explosion in list(self.explosions):
            if self.player.tile in explosion.tiles and self.player.id not in explosion.hit_actors:
                explosion.hit_actors.add(self.player.id)
                self.damage_player("explosion")
            for enemy in list(self.enemies):
                if enemy.alive and enemy.tile in explosion.tiles and enemy.id not in explosion.hit_actors:
                    explosion.hit_actors.add(enemy.id)
                    self.kill_enemy(enemy)
            if not explosion.update(dt):
                self.explosions.remove(explosion)

    def damage_player(self, source: str) -> None:
        if self.player.invulnerable > 0 or self.game.state != "playing":
            return
        if self.player.shield_time > 0:
            self.player.shield_time = 0
            self.player.invulnerable = 0.8
            self.notify("SHIELD ABSORBED HIT", 1.4, (120, 195, 255))
            self._spawn_debris(self.player.tile, (120, 195, 255), 14)
            return
        self.player.lives = max(0, self.player.lives - 1)
        self.level_damage_taken = True
        self.game.sound.play("hit")
        self.game.shake = max(self.game.shake, 10.0)
        self._spawn_debris(self.player.tile, COLORS["red"], 18)
        if self.player.lives <= 0:
            self.game.end_run(victory=False)
            return
        danger_tiles = set(self.danger_map)
        danger_tiles.update(tile for ex in self.explosions for tile in ex.tiles)
        safe = self.arena.nearest_safe_tile(self.player.spawn_tile, danger_tiles)
        self.player.pos = tile_center(safe)
        self.player.invulnerable = 2.0
        self.player.can_pass_bombs.clear()
        # Remove immediate unavoidable hazards around the respawn location.
        near = {tile for tile in danger_tiles if abs(tile[0] - safe[0]) + abs(tile[1] - safe[1]) <= 1}
        self.explosions = [ex for ex in self.explosions if not (ex.tiles & near)]
        self.bombs = [bomb for bomb in self.bombs if abs(bomb.tile[0] - safe[0]) + abs(bomb.tile[1] - safe[1]) > 1]
        self.notify(f"SYSTEM REBOOT — {self.player.lives} lives", 1.8, COLORS["red"])

    def kill_enemy(self, enemy: Enemy) -> None:
        if not enemy.alive:
            return
        enemy.alive = False
        if enemy in self.enemies:
            self.enemies.remove(enemy)
        self.add_score(enemy.score_value, combo_action=True, at=enemy.pos.copy())
        self.game.sound.play("enemy")
        self._spawn_debris(enemy.tile, ENEMY_COLORS[enemy.kind], 16)
        if not self.enemies:
            self.notify("REACTOR CLEAR — FIND THE EXIT", 2.2, COLORS["green"])
            self.arena.exit_active = self.arena.exit_revealed

    def _resolve_enemy_contacts(self) -> None:
        for enemy in self.enemies:
            if enemy.pos.distance_to(self.player.pos) < enemy.radius + self.player.radius - 2:
                self.damage_player("enemy")
                break

    def _maybe_spawn_powerup(self, tile: Tuple[int, int]) -> None:
        if tile == self.arena.exit_tile:
            return
        chance = (0.20 - min(self.level, 10) * 0.006) * DIFFICULTIES[self.difficulty]["powerup"]
        if self.rng.random() >= max(0.07, chance):
            return
        weighted = [
            "bomb", "bomb", "range", "range", "speed", "shield", "fuse",
            "remote" if self.level >= 3 else "shield",
            "kick" if self.level >= 4 else "speed",
            "life" if self.rng.random() < 0.18 else "shield",
        ]
        self.powerups.append(PowerUp(self.rng.choice(weighted), tile))

    def _collect_powerups(self) -> None:
        for powerup in list(self.powerups):
            if powerup.tile != self.player.tile:
                continue
            self.powerups.remove(powerup)
            kind = powerup.kind
            if kind == "bomb":
                self.player.bomb_capacity = min(6, self.player.bomb_capacity + 1)
            elif kind == "range":
                self.player.blast_range = min(7, self.player.blast_range + 1)
            elif kind == "speed":
                self.player.speed_level = min(5, self.player.speed_level + 1)
            elif kind == "life":
                self.player.lives = min(9, self.player.lives + 1)
            elif kind == "shield":
                self.player.shield_time = 25.0
            elif kind == "fuse":
                self.player.fuse_level = min(5, self.player.fuse_level + 1)
            elif kind == "remote":
                self.player.has_remote = True
            elif kind == "kick":
                self.player.has_kick = True
            _, color, message = POWERUP_INFO[kind]
            self.notify(message, 1.7, color)
            self.add_score(150, combo_action=False, at=self.player.pos.copy())
            self.game.sound.play("pickup")
            self._spawn_debris(powerup.tile, color, 12)

    def _spawn_debris(self, tile: Tuple[int, int], color: Tuple[int, int, int], count: int) -> None:
        center = tile_center(tile) + pygame.Vector2(BOARD_X, BOARD_Y)
        for _ in range(count):
            angle = self.rng.random() * math.tau
            speed = self.rng.uniform(55, 180)
            self.particles.append(Particle(center.copy(), pygame.Vector2(math.cos(angle), math.sin(angle)) * speed, color, self.rng.uniform(0.35, 0.8), 0.8, self.rng.uniform(2, 5), 120))

    def _spawn_explosion_particles(self, tiles: Iterable[Tuple[int, int]]) -> None:
        for tile in list(tiles)[:18]:
            self._spawn_debris(tile, self.rng.choice([COLORS["red"], COLORS["orange"], COLORS["yellow"]]), 3)

    def valid_enemy_moves(self, tile: Tuple[int, int], phase_blocks: bool, enemy_id: int) -> List[Tuple[int, int]]:
        result = []
        for dx, dy in DIRS:
            nxt = (tile[0] + dx, tile[1] + dy)
            if not self.enemy_tile_blocked(nxt, phase_blocks, enemy_id):
                result.append((dx, dy))
        return result

    def enemy_tile_blocked(self, tile: Tuple[int, int], phase_blocks: bool, enemy_id: int) -> bool:
        if self.arena.is_solid(tile, phase_blocks=phase_blocks):
            return True
        if self.bomb_at(tile):
            return True
        # Soft anti-stacking: reserve exact centered tiles occupied by another enemy.
        for enemy in self.enemies:
            if enemy.id != enemy_id and enemy.alive and enemy.tile == tile and enemy.pos.distance_to(tile_center(tile)) < 6:
                return True
        return False

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], avoid_danger: bool, phase_blocks: bool) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        queue = deque([start])
        previous: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        while queue:
            current = queue.popleft()
            for dx, dy in DIRS:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in previous:
                    continue
                if self.arena.is_solid(nxt, phase_blocks=phase_blocks) or self.bomb_at(nxt):
                    continue
                if avoid_danger and self.danger_map.get(nxt, 99.0) < 0.9:
                    continue
                previous[nxt] = current
                if nxt == goal:
                    path = [nxt]
                    while path[-1] != start:
                        path.append(previous[path[-1]])
                    path.reverse()
                    return path
                queue.append(nxt)
        return [start]

    def build_danger_map(self) -> Dict[Tuple[int, int], float]:
        effective = {bomb.id: max(0.0, bomb.timer) for bomb in self.bombs}
        changed = True
        for _ in range(len(self.bombs) + 1):
            if not changed:
                break
            changed = False
            for bomb in self.bombs:
                tiles = self._blast_tiles(bomb)
                for other in self.bombs:
                    if other.id != bomb.id and other.tile in tiles and effective[other.id] > effective[bomb.id]:
                        effective[other.id] = effective[bomb.id]
                        changed = True
        danger: Dict[Tuple[int, int], float] = {}
        for bomb in self.bombs:
            t = effective[bomb.id]
            for tile in self._blast_tiles(bomb):
                danger[tile] = min(danger.get(tile, 99.0), t)
        for explosion in self.explosions:
            for tile in explosion.tiles:
                danger[tile] = 0.0
        if self.reactor_warning > 0:
            for tile in self.reactor_active_tiles:
                danger[tile] = min(danger.get(tile, 99.0), self.reactor_warning)
        return danger

    def _update_challenge(self, dt: float) -> None:
        self.challenge_timer += dt
        if self.level == 6 and not self.reinforcement_done and self.level_elapsed > 18:
            self.reinforcement_done = True
            candidates = [tile for tile in self.arena.floor_tiles() if abs(tile[0] - self.player.tile[0]) + abs(tile[1] - self.player.tile[1]) > 7 and not self.bomb_at(tile)]
            self.rng.shuffle(candidates)
            for tile in candidates[:2]:
                self.enemies.append(HunterBot(tile, self.difficulty, self.level, self.rng))
            self.notify("REINFORCEMENTS DETECTED", 2.2, COLORS["red"])
        if self.level == 9:
            if self.reactor_warning > 0:
                self.reactor_warning -= dt
                if self.reactor_warning <= 0:
                    tiles = set(self.reactor_active_tiles)
                    if tiles:
                        ex = Explosion(self.explosion_counter, tiles, 0.7, 0.7)
                        self.explosion_counter += 1
                        self.explosions.append(ex)
                        self._spawn_explosion_particles(tiles)
                        self.game.sound.play("explode")
                        self.game.shake = max(self.game.shake, 8)
                    self.reactor_active_tiles.clear()
                    self.reactor_cycle = 7.5
            else:
                self.reactor_cycle -= dt
                if self.reactor_cycle <= 0:
                    pool = list(self.arena.unstable_tiles)
                    self.rng.shuffle(pool)
                    self.reactor_active_tiles = set(pool[:6])
                    self.reactor_warning = 1.6
                    self.notify("REACTOR SURGE — MOVE!", 1.6, COLORS["red"])

    def finish_level(self) -> None:
        time_bonus = max(0, int((105 - self.level_elapsed) * 15))
        no_hit_bonus = 1500 if not self.level_damage_taken else 0
        self.add_score(1000 + self.level * 250 + time_bonus + no_hit_bonus)
        self.game.sound.play("level")
        self.game.level_summary = {
            "level": self.level,
            "time": self.level_elapsed,
            "time_bonus": time_bonus,
            "no_hit_bonus": no_hit_bonus,
            "score": self.score,
        }
        if self.level >= 10:
            self.game.end_run(victory=True)
        else:
            self.game.state = "level_complete"

    def draw_bomb_previews(self, surface: pygame.Surface, time_s: float) -> None:
        """Show every bomb's wall-correct blast footprint before detonation."""
        if not self.bombs:
            return
        preview = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        for bomb in self.bombs:
            urgency = 1.0 - clamp(bomb.timer / 3.0, 0.0, 1.0)
            pulse = 0.5 + 0.5 * math.sin(time_s * (6.0 + urgency * 8.0) + bomb.id)
            fill_alpha = int(34 + urgency * 52 + pulse * 18)
            edge_alpha = int(115 + urgency * 105 + pulse * 25)
            fill_color = (255, int(72 + 70 * (1.0 - urgency)), 74, fill_alpha)
            edge_color = (255, 205, 82, min(255, edge_alpha))
            for x, y in self._blast_tiles(bomb):
                rect = pygame.Rect(x * TILE + 5, y * TILE + 5, TILE - 10, TILE - 10)
                pygame.draw.rect(preview, fill_color, rect, border_radius=8)
                pygame.draw.rect(preview, edge_color, rect, 2, border_radius=8)
                # The inner cross remains visible on both bright and dark images.
                cx, cy = rect.center
                pygame.draw.line(preview, (255, 238, 170, min(220, edge_alpha)), (rect.left + 8, cy), (rect.right - 8, cy), 1)
                pygame.draw.line(preview, (255, 238, 170, min(220, edge_alpha)), (cx, rect.top + 8), (cx, rect.bottom - 8), 1)
        surface.blit(preview, (BOARD_X, BOARD_Y))

    def draw(self, surface: pygame.Surface, fonts: Dict[str, pygame.font.Font], time_s: float) -> None:
        self.arena.draw(surface, time_s)
        self.draw_bomb_previews(surface, time_s)

        if DIFFICULTIES[self.difficulty]["danger_hints"]:
            hint = pygame.Surface((TILE - 8, TILE - 8), pygame.SRCALPHA)
            hint.fill((255, 60, 80, 50))
            for tile, t in self.danger_map.items():
                if t < 1.25:
                    surface.blit(hint, (BOARD_X + tile[0] * TILE + 4, BOARD_Y + tile[1] * TILE + 4))

        for tile in self.reactor_active_tiles:
            rect = pygame.Rect(BOARD_X + tile[0] * TILE + 5, BOARD_Y + tile[1] * TILE + 5, TILE - 10, TILE - 10)
            pygame.draw.rect(surface, COLORS["red"], rect, 4, border_radius=7)

        for powerup in self.powerups:
            powerup.draw(surface, fonts["small_bold"])
        for bomb in self.bombs:
            bomb.draw(surface, time_s)
        for enemy in self.enemies:
            enemy.draw(surface, time_s)
        self.player.draw(surface, time_s)
        for explosion in self.explosions:
            explosion.draw(surface, time_s)
        for particle in self.particles:
            particle.draw(surface)
        for floater in self.floaters:
            floater.draw(surface, fonts["tiny"])

        if self.level == 8:
            pulse = 0.5 + 0.5 * math.sin(self.level_elapsed * 0.7)
            alpha = int(45 + 145 * pulse)
            blackout = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            blackout.fill((0, 0, 8, alpha))
            # A soft circular opening keeps the player readable.
            mask = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            mask.fill((0, 0, 0, 0))
            pygame.draw.circle(mask, (0, 0, 0, max(0, alpha - 125)), (int(self.player.pos.x), int(self.player.pos.y)), 120)
            blackout.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
            surface.blit(blackout, (BOARD_X, BOARD_Y))

        self.draw_hud(surface, fonts)
        if self.notifications:
            text, _, color = self.notifications[0]
            panel = pygame.Rect(BOARD_X + 120, BOARD_Y + BOARD_H // 2 - 36, BOARD_W - 240, 72)
            overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
            overlay.fill((5, 10, 20, 215))
            surface.blit(overlay, panel)
            pygame.draw.rect(surface, color, panel, 2, border_radius=10)
            draw_text(surface, fonts["medium"], text, panel.center, color, "center", True)

    def draw_hud(self, surface: pygame.Surface, fonts: Dict[str, pygame.font.Font]) -> None:
        panel = pygame.Rect(HUD_X, BOARD_Y, HUD_W, BOARD_H)
        pygame.draw.rect(surface, COLORS["panel"], panel, border_radius=14)
        pygame.draw.rect(surface, (35, 85, 112), panel, 2, border_radius=14)
        draw_text(surface, fonts["medium"], "BOMBER-VISION", (HUD_X + 20, BOARD_Y + 18), COLORS["cyan"])
        draw_text(surface, fonts["small"], self.difficulty.upper(), (HUD_X + HUD_W - 18, BOARD_Y + 23), COLORS["magenta"], "topright")

        stats = [
            ("SCORE", f"{self.score:,}"),
            ("HIGH", f"{self.game.save.high_score(self.difficulty):,}"),
            ("LEVEL", f"{self.level} / 10"),
            ("LIVES", str(self.player.lives)),
            ("ENEMIES", str(len(self.enemies))),
            ("BOMBS", f"{len(self._player_bombs())} / {self.player.bomb_capacity}"),
            ("RANGE", str(self.player.blast_range)),
            ("COMBO", f"x{self.combo_multiplier}"),
        ]
        y = BOARD_Y + 72
        for label, value in stats:
            draw_text(surface, fonts["tiny"], label, (HUD_X + 20, y), COLORS["muted"])
            color = COLORS["yellow"] if label == "COMBO" and self.combo_multiplier > 1 else COLORS["white"]
            draw_text(surface, fonts["small_bold"], value, (HUD_X + HUD_W - 20, y - 2), color, "topright")
            y += 38

        pygame.draw.line(surface, (35, 69, 95), (HUD_X + 18, y + 2), (HUD_X + HUD_W - 18, y + 2), 1)
        y += 18
        draw_text(surface, fonts["tiny"], "UPGRADES", (HUD_X + 20, y), COLORS["cyan"])
        y += 28
        upgrades = []
        if self.player.has_remote:
            upgrades.append("REMOTE")
        if self.player.has_kick:
            upgrades.append("KICK")
        if self.player.fuse_level:
            upgrades.append(f"FUSE {self.player.fuse_level}")
        if self.player.speed_level:
            upgrades.append(f"SPEED {self.player.speed_level}")
        if not upgrades:
            upgrades = ["—"]
        draw_text(surface, fonts["small"], "  ·  ".join(upgrades), (HUD_X + 20, y), COLORS["white"])
        y += 38
        if self.player.shield_time > 0:
            draw_text(surface, fonts["small"], f"SHIELD  {self.player.shield_time:0.1f}s", (HUD_X + 20, y), (120, 195, 255))
            y += 32
        draw_text(surface, fonts["tiny"], "SPACE: bomb / remote", (HUD_X + 20, BOARD_Y + BOARD_H - 76), COLORS["muted"])
        draw_text(surface, fonts["tiny"], "P/ESC: pause    M: mute", (HUD_X + 20, BOARD_Y + BOARD_H - 50), COLORS["muted"])
        draw_text(surface, fonts["tiny"], "F11: fullscreen", (HUD_X + 20, BOARD_Y + BOARD_H - 25), COLORS["muted"])

# -----------------------------------------------------------------------------
# Main application and states
# -----------------------------------------------------------------------------

class Game:
    def __init__(self, smoke_test: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption("Bomber-Vision")
        self.save = SaveManager()
        self.fullscreen = bool(self.save.data["fullscreen"]) and not smoke_test
        self.windowed_size = (VIRTUAL_W, VIRTUAL_H)
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(self.windowed_size, flags)
        self.canvas = pygame.Surface((VIRTUAL_W, VIRTUAL_H)).convert()
        self.clock = pygame.time.Clock()
        self.sound = SoundManager(float(self.save.data["volume"]))
        self.running = True
        self.state = "title"
        self.previous_state = "title"
        self.difficulty = str(self.save.data["difficulty"])
        self.menu_index = 0
        self.menu_rects: List[pygame.Rect] = []
        self.session: Optional[GameSession] = None
        self.accumulator = 0.0
        self.time_s = 0.0
        self.intro_timer = 0.0
        self.level_summary: Dict[str, float] = {}
        self.new_high_score = False
        self.shake = 0.0
        self.mouse_virtual = pygame.Vector2(-999, -999)
        self.fonts = {
            "title": pygame.font.Font(None, 86),
            "large": pygame.font.Font(None, 58),
            "medium": pygame.font.Font(None, 34),
            "small_bold": pygame.font.Font(None, 27),
            "small": pygame.font.Font(None, 24),
            "tiny": pygame.font.Font(None, 20),
        }
        self.startup_background = self._load_random_startup_background()
        self._background_stars = [
            (random.randrange(VIRTUAL_W), random.randrange(VIRTUAL_H), random.uniform(0.3, 1.0))
            for _ in range(90)
        ]

    def _load_random_startup_background(self) -> Optional[pygame.Surface]:
        """Load one FrameVision startup image and crop it to the virtual canvas.

        Bomber-Vision lives in ``<FrameVision root>/helpers`` while the shared
        startup backgrounds live in ``<FrameVision root>/presets/startup``.
        Invalid or unsupported images are skipped so a bad file cannot prevent
        the game from starting.
        """
        helpers_dir = Path(__file__).resolve().parent
        startup_dir = helpers_dir.parent / "presets" / "startup"
        if not startup_dir.is_dir():
            return None

        supported = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".webp", ".tga", ".gif"}
        candidates = [
            path for path in startup_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported
        ]
        random.shuffle(candidates)

        for path in candidates:
            try:
                image = pygame.image.load(str(path)).convert()
                width, height = image.get_size()
                if width <= 0 or height <= 0:
                    continue

                # Cover the complete 1280x720 virtual canvas without stretching.
                scale = max(VIRTUAL_W / width, VIRTUAL_H / height)
                scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = pygame.transform.smoothscale(image, scaled_size)
                background = pygame.Surface((VIRTUAL_W, VIRTUAL_H)).convert()
                background.blit(
                    image,
                    ((VIRTUAL_W - scaled_size[0]) // 2, (VIRTUAL_H - scaled_size[1]) // 2),
                )
                # Darken once at load time instead of allocating an overlay on
                # every frame. The shared artwork remains visible behind the UI.
                shade = pygame.Surface((VIRTUAL_W, VIRTUAL_H), pygame.SRCALPHA)
                shade.fill((2, 5, 14, 118))
                background.blit(shade, (0, 0))
                return background
            except (pygame.error, OSError, ValueError):
                continue
        return None

    def start_new_game(self) -> None:
        self.session = GameSession(self, self.difficulty)
        self.state = "level_intro"
        self.intro_timer = 2.8 if self.session.challenge else 1.8
        self.new_high_score = False
        self.accumulator = 0.0
        self.sound.play("menu")

    def begin_next_level(self) -> None:
        if not self.session:
            return
        next_level = self.session.level + 1
        self.session.start_level(next_level, preserve_player=True)
        self.state = "level_intro"
        self.intro_timer = 2.8 if self.session.challenge else 1.8
        self.accumulator = 0.0

    def end_run(self, victory: bool) -> None:
        if not self.session:
            return
        self.new_high_score = self.save.submit_score(self.difficulty, self.session.score)
        self.state = "victory" if victory else "game_over"
        self.sound.play("level" if victory else "gameover")
        self.accumulator = 0.0

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        self.save.data["fullscreen"] = self.fullscreen
        self.save.save()

    def set_volume(self, value: float) -> None:
        self.sound.set_volume(value)
        self.save.data["volume"] = self.sound.volume
        self.save.save()

    def screen_to_virtual(self, pos: Tuple[int, int]) -> pygame.Vector2:
        sw, sh = self.screen.get_size()
        scale = min(sw / VIRTUAL_W, sh / VIRTUAL_H)
        draw_w, draw_h = VIRTUAL_W * scale, VIRTUAL_H * scale
        ox, oy = (sw - draw_w) / 2, (sh - draw_h) / 2
        return pygame.Vector2((pos[0] - ox) / scale, (pos[1] - oy) / scale)

    def activate_menu(self) -> None:
        if self.menu_index == 0:
            self.start_new_game()
        elif self.menu_index == 1:
            self.state = "instructions"
            self.sound.play("menu")
        elif self.menu_index == 2:
            self.cycle_difficulty(1)
        elif self.menu_index == 3:
            self.state = "high_scores"
            self.sound.play("menu")
        elif self.menu_index == 4:
            self.running = False

    def cycle_difficulty(self, direction: int) -> None:
        values = list(DIFFICULTIES)
        index = values.index(self.difficulty)
        self.difficulty = values[(index + direction) % len(values)]
        self.save.data["difficulty"] = self.difficulty
        self.save.save()
        self.sound.play("menu")

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.windowed_size = (max(640, event.w), max(360, event.h))
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            return
        if event.type == pygame.MOUSEMOTION:
            self.mouse_virtual = self.screen_to_virtual(event.pos)
            if self.state == "title":
                for i, rect in enumerate(self.menu_rects):
                    if rect.collidepoint(self.mouse_virtual):
                        self.menu_index = i
                        break
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse_virtual = self.screen_to_virtual(event.pos)
            if self.state == "title":
                for i, rect in enumerate(self.menu_rects):
                    if rect.collidepoint(self.mouse_virtual):
                        self.menu_index = i
                        self.activate_menu()
                        return
            elif self.state in {"instructions", "high_scores"}:
                self.state = "title"
                self.sound.play("menu")
            elif self.state == "level_complete":
                self.begin_next_level()
            return
        if event.type != pygame.KEYDOWN:
            return

        if event.key == pygame.K_F11:
            self.toggle_fullscreen()
            return
        if event.key == pygame.K_m:
            self.sound.toggle_mute()
            return
        if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.set_volume(self.sound.volume + 0.05)
            return
        if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.set_volume(self.sound.volume - 0.05)
            return

        if self.state == "title":
            if event.key in (pygame.K_UP, pygame.K_w):
                self.menu_index = (self.menu_index - 1) % 5
                self.sound.play("menu")
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.menu_index = (self.menu_index + 1) % 5
                self.sound.play("menu")
            elif event.key in (pygame.K_LEFT, pygame.K_a) and self.menu_index == 2:
                self.cycle_difficulty(-1)
            elif event.key in (pygame.K_RIGHT, pygame.K_d) and self.menu_index == 2:
                self.cycle_difficulty(1)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.activate_menu()
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif self.state in {"instructions", "high_scores"}:
            if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.state = "title"
                self.sound.play("menu")
        elif self.state == "level_intro":
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.intro_timer = 0
        elif self.state == "playing":
            if event.key == pygame.K_SPACE and self.session:
                self.session.place_or_remote()
            elif event.key in (pygame.K_p, pygame.K_ESCAPE):
                self.state = "paused"
                self.sound.play("menu")
        elif self.state == "paused":
            if event.key in (pygame.K_p, pygame.K_ESCAPE):
                self.state = "playing"
                self.sound.play("menu")
            elif event.key == pygame.K_q:
                self.state = "title"
                self.session = None
        elif self.state == "level_complete":
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.begin_next_level()
            elif event.key == pygame.K_ESCAPE:
                self.state = "title"
                self.session = None
        elif self.state in {"game_over", "victory"}:
            if event.key == pygame.K_r:
                self.start_new_game()
            elif event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                self.state = "title"
                self.session = None

    def movement_vector(self) -> pygame.Vector2:
        keys = pygame.key.get_pressed()
        return pygame.Vector2(
            int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a]),
            int(keys[pygame.K_DOWN] or keys[pygame.K_s]) - int(keys[pygame.K_UP] or keys[pygame.K_w]),
        )

    def update(self, frame_dt: float) -> None:
        self.time_s += frame_dt
        self.shake = max(0.0, self.shake - 22 * frame_dt)
        if self.state == "level_intro":
            self.intro_timer -= frame_dt
            if self.intro_timer <= 0:
                self.state = "playing"
                self.accumulator = 0.0
            return
        if self.state != "playing" or not self.session:
            return

        self.accumulator = min(self.accumulator + frame_dt, 0.25)
        movement = self.movement_vector()
        while self.accumulator >= FIXED_DT and self.state == "playing":
            self.session.update(FIXED_DT, movement)
            self.accumulator -= FIXED_DT

    def draw_background(self) -> None:
        if self.startup_background is not None:
            self.canvas.blit(self.startup_background, (0, 0))
        else:
            self.canvas.fill(COLORS["bg"])
        for x, y, speed in self._background_stars:
            yy = (y + self.time_s * 12 * speed) % VIRTUAL_H
            brightness = int(65 + 120 * speed)
            pygame.draw.circle(self.canvas, (brightness // 2, brightness, brightness), (x, int(yy)), 1)
        grid = 48
        offset = int(self.time_s * 7) % grid
        for x in range(-grid, VIRTUAL_W + grid, grid):
            pygame.draw.line(self.canvas, (8, 23, 40), (x + offset, 0), (x + offset - 140, VIRTUAL_H), 1)
        pygame.draw.rect(self.canvas, (11, 25, 44), (0, 0, VIRTUAL_W, 82))
        pygame.draw.line(self.canvas, COLORS["cyan2"], (0, 81), (VIRTUAL_W, 81), 2)
        draw_text(self.canvas, self.fonts["tiny"], "BV // BOMBER GRID SIMULATION", (26, 28), COLORS["muted"])
        audio = "MUTED" if self.sound.muted else f"VOL {int(self.sound.volume * 100)}%"
        draw_text(self.canvas, self.fonts["tiny"], audio, (VIRTUAL_W - 26, 28), COLORS["muted"], "topright")

    def draw_title(self) -> None:
        glow = pygame.Surface((VIRTUAL_W, VIRTUAL_H), pygame.SRCALPHA)
        pygame.draw.circle(glow, (0, 180, 255, 35), (310, 340), 260)
        pygame.draw.circle(glow, (255, 30, 180, 28), (955, 390), 300)
        self.canvas.blit(glow, (0, 0))
        draw_text(self.canvas, self.fonts["title"], "BOMBER-VISION", (VIRTUAL_W // 2, 140), COLORS["cyan"], "center", True)
        draw_text(self.canvas, self.fonts["small"], "DESTRUCT  •  EVADE  •  SURVIVE", (VIRTUAL_W // 2, 200), COLORS["magenta"], "center")

        labels = [
            "START GAME",
            "INSTRUCTIONS",
            f"DIFFICULTY  ‹  {self.difficulty.upper()}  ›",
            "HIGH SCORES",
            "QUIT",
        ]
        self.menu_rects = []
        y = 272
        for i, label in enumerate(labels):
            rect = pygame.Rect(VIRTUAL_W // 2 - 230, y, 460, 54)
            self.menu_rects.append(rect)
            selected = i == self.menu_index
            color = COLORS["cyan"] if selected else COLORS["muted"]
            bg = (18, 41, 62) if selected else (10, 18, 31)
            pygame.draw.rect(self.canvas, bg, rect, border_radius=9)
            pygame.draw.rect(self.canvas, color, rect, 2 if selected else 1, border_radius=9)
            if selected:
                pygame.draw.polygon(self.canvas, COLORS["magenta"], [(rect.left + 18, rect.centery), (rect.left + 28, rect.centery - 7), (rect.left + 28, rect.centery + 7)])
            draw_text(self.canvas, self.fonts["medium"], label, rect.center, color, "center")
            y += 66
        draw_text(self.canvas, self.fonts["tiny"], "UP/DOWN select   LEFT/RIGHT difficulty   Enter confirm   F11 fullscreen", (VIRTUAL_W // 2, 650), COLORS["muted"], "center")

    def draw_instructions(self) -> None:
        draw_text(self.canvas, self.fonts["large"], "OPERATIONS MANUAL", (VIRTUAL_W // 2, 120), COLORS["cyan"], "center")
        left = 120
        right = 680
        y = 180
        sections = [
            (left, y, "CONTROLS", [
                "WASD / Arrows — move",
                "Space — place bomb / remote detonate",
                "P or Esc — pause",
                "M — mute     +/- — volume",
                "F11 — fullscreen",
            ]),
            (left, 390, "OBJECTIVE", [
                "Destroy blocks and defeat every enemy.",
                "Reveal the exit hidden beneath a block.",
                "The exit activates when the arena is clear.",
                "Survive all 10 escalating reactor sectors.",
            ]),
            (right, y, "BOMB RULES", [
                "Blasts stop at solid walls.",
                "Destructible blocks stop one blast arm.",
                "Bombs trigger chain reactions.",
                "Walk away from a newly placed bomb freely.",
                "Kick and remote upgrades alter bomb control.",
            ]),
            (right, 390, "POWER UPS", [
                "B capacity   R blast range   S speed",
                "+ life       O shield        F fuse",
                "D remote     K bomb kick",
                "Easy mode displays near-term danger tiles.",
            ]),
        ]
        for x, sy, heading, lines in sections:
            draw_text(self.canvas, self.fonts["medium"], heading, (x, sy), COLORS["magenta"])
            yy = sy + 42
            for line in lines:
                draw_text(self.canvas, self.fonts["small"], line, (x, yy), COLORS["white"])
                yy += 31
        draw_text(self.canvas, self.fonts["small"], "Press Enter, Space, Esc, or click to return", (VIRTUAL_W // 2, 665), COLORS["muted"], "center")

    def draw_high_scores(self) -> None:
        draw_text(self.canvas, self.fonts["large"], "HIGH SCORES", (VIRTUAL_W // 2, 145), COLORS["cyan"], "center")
        y = 250
        for difficulty in DIFFICULTIES:
            rect = pygame.Rect(VIRTUAL_W // 2 - 270, y, 540, 76)
            pygame.draw.rect(self.canvas, COLORS["panel"], rect, border_radius=12)
            pygame.draw.rect(self.canvas, COLORS["cyan2"], rect, 2, border_radius=12)
            draw_text(self.canvas, self.fonts["medium"], difficulty.upper(), (rect.left + 28, rect.centery), COLORS["magenta"], "midleft")
            draw_text(self.canvas, self.fonts["large"], f"{self.save.high_score(difficulty):,}", (rect.right - 28, rect.centery), COLORS["white"], "midright")
            y += 96
        draw_text(self.canvas, self.fonts["small"], "Press Enter, Space, Esc, or click to return", (VIRTUAL_W // 2, 650), COLORS["muted"], "center")

    def draw_gameplay_state(self) -> None:
        if not self.session:
            return
        self.session.draw(self.canvas, self.fonts, self.time_s)
        if self.state == "level_intro":
            self.draw_overlay(210)
            draw_text(self.canvas, self.fonts["large"], f"SECTOR {self.session.level}", (VIRTUAL_W // 2, 280), COLORS["cyan"], "center", True)
            if self.session.challenge:
                draw_text(self.canvas, self.fonts["medium"], self.session.challenge, (VIRTUAL_W // 2, 342), COLORS["yellow"], "center")
            else:
                draw_text(self.canvas, self.fonts["medium"], "PURGE HOSTILES. LOCATE THE EXIT.", (VIRTUAL_W // 2, 342), COLORS["white"], "center")
            draw_text(self.canvas, self.fonts["small"], "Enter or Space to deploy now", (VIRTUAL_W // 2, 400), COLORS["muted"], "center")
        elif self.state == "paused":
            self.draw_overlay(205)
            draw_text(self.canvas, self.fonts["large"], "PAUSED", (VIRTUAL_W // 2, 286), COLORS["cyan"], "center")
            draw_text(self.canvas, self.fonts["medium"], "P / Esc — resume", (VIRTUAL_W // 2, 360), COLORS["white"], "center")
            draw_text(self.canvas, self.fonts["small"], "Q — abandon run and return to title", (VIRTUAL_W // 2, 410), COLORS["muted"], "center")
        elif self.state == "level_complete":
            self.draw_overlay(218)
            summary = self.level_summary
            draw_text(self.canvas, self.fonts["large"], "SECTOR CLEARED", (VIRTUAL_W // 2, 205), COLORS["green"], "center", True)
            lines = [
                f"Clear time: {summary.get('time', 0):.1f}s",
                f"Time bonus: +{int(summary.get('time_bonus', 0)):,}",
                f"No-damage bonus: +{int(summary.get('no_hit_bonus', 0)):,}",
                f"Total score: {int(summary.get('score', 0)):,}",
            ]
            y = 290
            for line in lines:
                draw_text(self.canvas, self.fonts["medium"], line, (VIRTUAL_W // 2, y), COLORS["white"], "center")
                y += 46
            draw_text(self.canvas, self.fonts["small"], "Enter, Space, or click — next sector", (VIRTUAL_W // 2, 505), COLORS["cyan"], "center")

    def draw_overlay(self, alpha: int) -> None:
        overlay = pygame.Surface((VIRTUAL_W, VIRTUAL_H), pygame.SRCALPHA)
        overlay.fill((2, 5, 12, alpha))
        self.canvas.blit(overlay, (0, 0))

    def draw_end_screen(self, victory: bool) -> None:
        self.draw_overlay(230)
        title = "REACTOR STABILIZED" if victory else "SYSTEM FAILURE"
        color = COLORS["green"] if victory else COLORS["red"]
        draw_text(self.canvas, self.fonts["large"], title, (VIRTUAL_W // 2, 205), color, "center", True)
        score = self.session.score if self.session else 0
        level = self.session.level if self.session else 0
        draw_text(self.canvas, self.fonts["medium"], f"Final score: {score:,}", (VIRTUAL_W // 2, 300), COLORS["white"], "center")
        draw_text(self.canvas, self.fonts["medium"], f"Sector reached: {level} / 10", (VIRTUAL_W // 2, 348), COLORS["white"], "center")
        draw_text(self.canvas, self.fonts["small"], f"Difficulty: {self.difficulty}", (VIRTUAL_W // 2, 396), COLORS["muted"], "center")
        if self.new_high_score:
            draw_text(self.canvas, self.fonts["large"], "NEW HIGH SCORE", (VIRTUAL_W // 2, 460), COLORS["yellow"], "center")
        draw_text(self.canvas, self.fonts["small"], "R — new run     Enter / Esc — title screen", (VIRTUAL_W // 2, 565), COLORS["cyan"], "center")

    def draw(self) -> None:
        self.draw_background()
        if self.state == "title":
            self.draw_title()
        elif self.state == "instructions":
            self.draw_instructions()
        elif self.state == "high_scores":
            self.draw_high_scores()
        elif self.state in {"playing", "paused", "level_intro", "level_complete"}:
            self.draw_gameplay_state()
        elif self.state == "game_over":
            if self.session:
                self.session.draw(self.canvas, self.fonts, self.time_s)
            self.draw_end_screen(False)
        elif self.state == "victory":
            if self.session:
                self.session.draw(self.canvas, self.fonts, self.time_s)
            self.draw_end_screen(True)

        frame = self.canvas
        if self.shake > 0 and self.state in {"playing", "level_complete", "game_over", "victory"}:
            shaken = pygame.Surface((VIRTUAL_W, VIRTUAL_H))
            shaken.fill(COLORS["bg"])
            offset = (random.randint(-int(self.shake), int(self.shake)), random.randint(-int(self.shake), int(self.shake)))
            shaken.blit(self.canvas, offset)
            frame = shaken
        self.present(frame)

    def present(self, frame: pygame.Surface) -> None:
        sw, sh = self.screen.get_size()
        scale = min(sw / VIRTUAL_W, sh / VIRTUAL_H)
        size = (max(1, int(VIRTUAL_W * scale)), max(1, int(VIRTUAL_H * scale)))
        scaled = pygame.transform.smoothscale(frame, size)
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, ((sw - size[0]) // 2, (sh - size[1]) // 2))
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            frame_dt = min(self.clock.tick(FPS) / 1000.0, 0.1)
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(frame_dt)
            self.draw()
        self.save.save()
        pygame.quit()


def run_smoke_test() -> None:
    """Headless sanity pass used during development and harmless for users."""
    game = Game(smoke_test=True)
    game.start_new_game()
    game.state = "playing"
    assert game.session is not None
    session = game.session
    # Basic placement, collision, detonation, danger map, restart and level reset.
    session.place_or_remote()
    assert len(session.bombs) == 1
    session.place_or_remote()  # Same tile and capacity must not stack.
    assert len(session.bombs) == 1
    session.bombs[0].timer = 0.0
    session.update(FIXED_DT, pygame.Vector2())
    assert session.explosions
    for _ in range(120):
        session.update(FIXED_DT, pygame.Vector2())
    assert not session.bombs
    session.start_level(2, preserve_player=True)
    assert session.level == 2 and not session.bombs and not session.explosions
    game.save.save()
    pygame.quit()
    print("Bomber-Vision smoke test passed")


def main() -> None:
    if "--smoke-test" in sys.argv:
        run_smoke_test()
    else:
        Game().run()


if __name__ == "__main__":
    main()
