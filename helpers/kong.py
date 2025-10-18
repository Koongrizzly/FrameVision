#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donkey Kong–style mini-clone built with Pygame.

Patch history:
- 2025-10-18 (v1): Reward animation after collecting top reward; "START LEVEL N" intro card; debug spam off by default.
- 2025-10-18 (v2): Button bar (Start / Pause / Exit / Help / High Scores).
                   F1 opens Help, Escape closes Help/High Scores.
                   High scores {date, score, name} saved to /presets/setsave/kong/kongsave.json.
                   Opening Help or High Scores during play auto-pauses and resumes when closed.
"""

import os
import sys
import math
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# --- Pygame bootstrap ---------------------------------------------------------------------------
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning)
_warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")
try:
    import pygame
except Exception as e:
    raise SystemExit("Pygame is required. Install with: pip install pygame") from e


# --- Constants ----------------------------------------------------------------------------------
WIDTH, HEIGHT = 900, 640
FPS = 60

GRAVITY = 0.50
PLAYER_SPEED = 3.2
PLAYER_JUMP_VELOCITY = -10.0
BARREL_BASE_SPEED = 2.0
BARREL_SPAWN_INTERVAL = 2.2  # seconds at level 1 (scales down with level)
BARREL_RADIUS = 22  # visual radius of barrel body (thumbnail sits inside)

FONT_NAME = "freesansbold.ttf"  # bundled with pygame

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GRAY   = (70,  70,  70)
LIGHT  = (200, 200, 200)
BROWN  = (130, 82,  1)
WOOD   = (160, 110, 50)
RED    = (220, 60,  60)
BLUE   = (60,  120, 220)

# Player profile (used for highscores)
PLAYER_NAME = os.getenv("KONG_PLAYER", "Player")

# --- Paths (relative to this file so it works inside /helpers/) ---------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # go from /helpers/kong.py → project ROOT
STARTUP_DIR = ROOT / "presets" / "startup"
THUMBS_DIR  = ROOT / "presets" / "setsave" / "thumbs" / "kong"

# Assets directory for custom player sprite
ASSETS_DIR = ROOT / "assets"
PLAYER_SPRITE_PATH = ASSETS_DIR / "player1.png"
# --- Sound assets ------------------------------------------------------------------------------
POP_WAV_PATH = ASSETS_DIR / "reward.mp3"           # reward sound
HURT_MP3_PATH = ASSETS_DIR / "hurt.mp3"         # player hit by barrel
GAME_OVER_MP3_PATH = ASSETS_DIR / "game_over.mp3"  # game over sound

class SFX:
    def __init__(self, debug: bool=False):
        self.debug = debug
        self.enabled = False
        self.sounds = {}
        self.volumes = {}
        self.muted = False
        try:
            # Initialize mixer (safe to call after pygame.init())
            pygame.mixer.init()
            self.enabled = True
        except Exception as e:
            if self.debug:
                print(f"[Kong][SFX] Audio disabled (mixer init failed): {e}")
            self.enabled = False

        if self.enabled:
            self._load("reward", POP_WAV_PATH)
            self._load("hurt", HURT_MP3_PATH)
            self._load("game_over", GAME_OVER_MP3_PATH)

    def _load(self, name: str, path: Path):
        try:
            if path.exists():
                self.sounds[name] = pygame.mixer.Sound(str(path))
                self.volumes[name] = self.sounds[name].get_volume()
                if self.debug:
                    print(f"[Kong][SFX] Loaded: {path}")
            else:
                if self.debug:
                    print(f"[Kong][SFX] Missing asset: {path}")
        except Exception as e:
            if self.debug:
                print(f"[Kong][SFX] Failed to load {path}: {e}")

    def play(self, name: str):
        try:
            if not self.enabled or self.muted:
                return
            snd = self.sounds.get(name)
            if snd is not None:
                snd.play()
        except Exception as e:
            if self.debug:
                print(f"[Kong][SFX] Play error for '{name}': {e}")

    def set_muted(self, muted: bool):
        self.muted = bool(muted)
        try:
            for n, snd in self.sounds.items():
                vol = 0.0 if self.muted else self.volumes.get(n, 1.0)
                try:
                    snd.set_volume(vol)
                except Exception:
                    pass
        except Exception:
            pass

    def toggle(self):
        self.set_muted(not self.muted)
   # user-provided transparent PNG

# Reward logo config
REWARD_W, REWARD_H = 100, 60
REWARD_RADIUS = 16

# High score file
SAVE_DIR = ROOT / "presets" / "setsave" / "kong"
SAVE_PATH = SAVE_DIR / "kongsave.json"


# --- Utils --------------------------------------------------------------------------------------
def ensure_dirs():
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def pick_random_jpg(source_dir: Path) -> Optional[Path]:
    if not source_dir.exists():
        return None
    exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
    jpgs = [p for p in source_dir.iterdir() if p.suffix in exts and p.is_file()]
    if not jpgs:
        return None
    return random.choice(jpgs)


def find_numbered_logos() -> List[Path]:
    """Return a list of available logo_{n}.jpg/.jpeg files in STARTUP_DIR sorted by number n."""
    if not STARTUP_DIR.exists():
        return []
    cands = []
    for p in STARTUP_DIR.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.startswith("logo_") and name.endswith((".jpg", ".jpeg"))):
            continue
        try:
            stem = p.stem  # 'logo_12'
            num_part = stem.split("_", 1)[1]
            n = int(num_part)
        except Exception:
            continue
        cands.append((n, p))
    cands.sort(key=lambda t: t[0])
    return [p for _, p in cands]


def path_for_logo_level(level: int) -> Optional[Path]:
    """Prefer exact match logo_{level}.jpg/.jpeg. If not found, cycle through available logo_* files."""
    exact_jpg = STARTUP_DIR / f"logo_{level}.jpg"
    exact_jpeg = STARTUP_DIR / f"logo_{level}.jpeg"
    if exact_jpg.exists():
        return exact_jpg
    if exact_jpeg.exists():
        return exact_jpeg
    logos = find_numbered_logos()
    if not logos:
        return None
    idx = (level - 1) % len(logos)
    return logos[idx]


def make_circular_thumbnail_pygame(src_path: Path, size: int) -> pygame.Surface:
    """Load image, crop-to-square center, scale to (size, size), apply circular alpha mask."""
    img = pygame.image.load(str(src_path)).convert_alpha()
    w, h = img.get_width(), img.get_height()
    side = min(w, h)
    crop_rect = pygame.Rect((w - side)//2, (h - side)//2, side, side)
    square = pygame.Surface((side, side), pygame.SRCALPHA)
    square.blit(img, (0, 0), crop_rect)
    thumb = pygame.transform.smoothscale(square, (size, size)).convert_alpha()
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (size//2, size//2), size//2)
    thumb.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return thumb


def make_rounded_rect_thumbnail_pygame(src_path: Path, out_w: int, out_h: int, radius: int) -> pygame.Surface:
    """Load an image, cover-fit to (out_w, out_h), center-crop, and apply a rounded-rectangle alpha mask."""
    img = pygame.image.load(str(src_path)).convert_alpha()
    w, h = img.get_width(), img.get_height()
    scale = max(out_w / w, out_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    scaled = pygame.transform.smoothscale(img, new_size).convert_alpha()
    x = (scaled.get_width() - out_w) // 2
    y = (scaled.get_height() - out_h) // 2
    crop = pygame.Surface((out_w, out_h), pygame.SRCALPHA)
    crop.blit(scaled, (0, 0), pygame.Rect(x, y, out_w, out_h))
    mask = pygame.Surface((out_w, out_h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), pygame.Rect(0, 0, out_w, out_h), border_radius=radius)
    crop.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return crop


def save_surface_png(surface: pygame.Surface, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def generate_level_thumbnail(level_index: int, size: int = 44) -> Tuple[pygame.Surface, Optional[Path]]:
    """Create a circular sticker for barrels from a random JPG in STARTUP_DIR; on failure, use placeholder."""
    ensure_dirs()
    src = pick_random_jpg(STARTUP_DIR)
    out_path = THUMBS_DIR / f"thumb_level_{level_index:03d}.png"

    if src is None:
        size = max(32, size)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (220, 220, 220, 255), (size//2, size//2), size//2)
        pygame.draw.circle(surf, (90, 90, 90, 255), (size//2, size//2), size//2, 3)
        font = pygame.font.Font(FONT_NAME, max(14, size//2))
        q = font.render("?", True, (90, 90, 90))
        surf.blit(q, q.get_rect(center=(size//2, size//2)))
        save_surface_png(surf, out_path)
        return surf, out_path

    try:
        surf = make_circular_thumbnail_pygame(src, size=size)
        save_surface_png(surf, out_path)
        return surf, out_path
    except Exception:
        size = max(32, size)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (220, 220, 220, 255), (size//2, size//2), size//2)
        pygame.draw.circle(surf, (200, 50, 50, 255), (size//2, size//2), size//2, 3)
        font = pygame.font.Font(FONT_NAME, max(12, size//3))
        t = font.render("ERR", True, (200, 50, 50))
        surf.blit(t, t.get_rect(center=(size//2, size//2)))
        save_surface_png(surf, out_path)
        return surf, out_path



# --- Level background overlay (20% alpha from /presets/startup/) -------------------------------
def load_random_background_overlay() -> Tuple[Optional[pygame.Surface], Optional[Path]]:
    """
    Choose a random JPG from STARTUP_DIR, scale-to-cover (WIDTH x HEIGHT),
    center-crop, and set ~20% opacity so it appears as a faint overlay on top
    of the game (but below modal overlays like Help/Game Over).
    Returns (surface, source_path) or (None, None) on failure.
    """
    src = pick_random_jpg(STARTUP_DIR)
    if src is None:
        return None, None
    try:
        img = pygame.image.load(str(src)).convert()
        w, h = img.get_width(), img.get_height()
        if w <= 0 or h <= 0:
            return None, src
        scale = max(WIDTH / w, HEIGHT / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        scaled = pygame.transform.smoothscale(img, new_size).convert()
        cx = max(0, (scaled.get_width()  - WIDTH)  // 2)
        cy = max(0, (scaled.get_height() - HEIGHT) // 2)
        out = pygame.Surface((WIDTH, HEIGHT)).convert()
        out.blit(scaled, (0, 0), pygame.Rect(cx, cy, WIDTH, HEIGHT))
        out.set_alpha(51)  # ~20% of 255
        return out, src
    except Exception:
        return None, src

# Custom player sprite ---------------------------------------------------------------------------
def load_player_sprite(debug: bool=False) -> Optional[pygame.Surface]:
    try:
        if PLAYER_SPRITE_PATH.exists():
            img = pygame.image.load(str(PLAYER_SPRITE_PATH)).convert_alpha()
            return img
        else:
            if debug:
                print(f"[Kong] No custom player sprite found at {PLAYER_SPRITE_PATH}. Using default box player.")
            return None
    except Exception as e:
        if debug:
            print(f"[Kong] Failed to load player sprite {PLAYER_SPRITE_PATH}: {e}")
        return None


# Reward logo loader -----------------------------------------------------------------------------
def load_reward_logo_for_level(level: int) -> Tuple[pygame.Surface, Optional[Path]]:
    target = path_for_logo_level(level)
    if target is None:
        surf = pygame.Surface((REWARD_W, REWARD_H), pygame.SRCALPHA)
        pygame.draw.rect(surf, (230, 230, 230, 255), pygame.Rect(0, 0, REWARD_W, REWARD_H), border_radius=REWARD_RADIUS)
        pygame.draw.rect(surf, (150, 150, 150, 255), pygame.Rect(0, 0, REWARD_W, REWARD_H), width=3, border_radius=REWARD_RADIUS)
        font = pygame.font.Font(FONT_NAME, 18)
        t = font.render("NO LOGO", True, (90, 90, 90))
        surf.blit(t, t.get_rect(center=(REWARD_W//2, REWARD_H//2)))
        return surf, None

    try:
        surf = make_rounded_rect_thumbnail_pygame(target, REWARD_W, REWARD_H, REWARD_RADIUS)
        return surf, target
    except Exception as e:
        surf = pygame.Surface((REWARD_W, REWARD_H), pygame.SRCALPHA)
        pygame.draw.rect(surf, (250, 230, 230, 255), pygame.Rect(0, 0, REWARD_W, REWARD_H), border_radius=REWARD_RADIUS)
        pygame.draw.rect(surf, (200, 60, 60, 255), pygame.Rect(0, 0, REWARD_W, REWARD_H), width=3, border_radius=REWARD_RADIUS)
        font = pygame.font.Font(FONT_NAME, 16)
        t = font.render("ERR", True, (200, 60, 60))
        surf.blit(t, t.get_rect(center=(REWARD_W//2, REWARD_H//2)))
        return surf, target


# --- Level geometry -----------------------------------------------------------------------------
@dataclass
class Platform:
    x1: int
    x2: int
    y: int
    slope_dir: int  # -1 = slopes left (barrels roll left), +1 = slopes right

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(min(self.x1, self.x2), self.y-6, abs(self.x2-self.x1), 12)


@dataclass
class Ladder:
    x: int
    y_top: int
    y_bottom: int

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x - 14, self.y_top, 28, self.y_bottom - self.y_top)

    @property
    def interact_rect(self) -> pygame.Rect:
        pad = 24
        return pygame.Rect(self.x - 12, self.y_top - pad, 24, (self.y_bottom - self.y_top) + pad*2)


def build_level(level: int) -> Tuple[List[Platform], List[Ladder]]:
    rng = random.Random(1000 + level)  # deterministic per level number
    margin = 60
    gap = 100
    top = 90
    plat_count = 5

    platforms: List[Platform] = []
    ladders: List[Ladder] = []

    for i in range(plat_count):
        y = HEIGHT - 80 - i*gap
        if i % 2 == 0:
            x1, x2 = margin, WIDTH - margin
            slope = +1
        else:
            x1, x2 = WIDTH - margin, margin
            slope = -1
        off = rng.randint(-30, 30)
        platforms.append(Platform(x1 + off, x2 + off, y, slope))

    platforms.append(Platform(margin, WIDTH - margin, top, +1))

    def pick_spaced_xs(min_x: int, max_x: int, k: int, min_sep: int, avoid: list) -> list:
        xs = []
        tries = 0
        while len(xs) < k and tries < 64:
            x = rng.randint(min_x, max_x)
            if all(abs(x - xi) >= min_sep for xi in xs) and all(abs(x - ax) >= 80 for ax in avoid):
                xs.append(x)
            tries += 1
        if len(xs) < k:
            step = (max_x - min_x) / (k + 1)
            xs = [int(min_x + step*(i+1)) for i in range(k)]
        return sorted(xs)

    prev_xs = []
    for i in range(len(platforms)-1):
        p_low = platforms[i]
        p_high = platforms[i+1]
        px1 = min(p_low.x1, p_low.x2) + 30
        px2 = max(p_low.x1, p_low.x2) - 30

        if level <= 3:
            for j in range(2):
                t = (j + 1) / 3.0
                x = int(px1 + t * (px2 - px1))
                x += rng.randint(-20, 20)
                y_top = p_high.y
                y_bottom = p_low.y
                ladders.append(Ladder(x, y_top, y_bottom))
            prev_xs = [ladders[-1].x, ladders[-2].x]
        else:
            min_sep = 150
            xs = pick_spaced_xs(px1, px2, k=2, min_sep=min_sep, avoid=prev_xs)
            for x in xs:
                y_top = p_high.y
                y_bottom = p_low.y
                ladders.append(Ladder(x, y_top, y_bottom))
            prev_xs = xs

    return platforms, ladders


# --- Entities -----------------------------------------------------------------------------------
class Player:
    def __init__(self, x: float, y: float, sprite: Optional[pygame.Surface] = None):
        self.w = 50
        self.h = 70
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.on_ladder = False
        self.invuln_time = 0.0
        self.coyote = 0.0
        self.last_platform_y: Optional[int] = None
        self.jumping: bool = False
        self.jump_lock_y: Optional[int] = None
        self.jump_was_down: bool = False
        self.color = BLUE

        # Ladder lock: restricts auto-align to the ladder you started on
        self.ladder_lock_x = None  # type: Optional[int]
        self.ladder_last_x = None  # remember last ladder x for strict climbing
        self.up_was_down = False
        self.facing = 1
        self.sprite_original = sprite
        self.sprite_right = None
        self.sprite_left = None
        if sprite is not None:
            try:
                scaled = pygame.transform.smoothscale(sprite, (self.w, self.h)).convert_alpha()
                self.sprite_right = scaled
                self.sprite_left = pygame.transform.flip(scaled, True, False)
            except Exception as e:
                self.sprite_right = None
                self.sprite_left = None

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def update(self, keys, platforms: List[Platform], ladders: List[Ladder], dt: float):
        speed = PLAYER_SPEED
        self.vx = 0.0
        jump_down = keys[pygame.K_SPACE]
        up_down = keys[pygame.K_UP]
        down_down = keys[pygame.K_DOWN]

        # Horizontal movement
        if keys[pygame.K_LEFT]:
            self.vx = -speed
        if keys[pygame.K_RIGHT]:
            self.vx = speed
        if self.vx < 0:
            self.facing = -1
        elif self.vx > 0:
            self.facing = +1


        # Determine if we are overlapping any ladder interaction area
        on_base_floor = (self.rect.bottom >= HEIGHT - 1)

        ladder_candidates = [lad for lad in ladders if self.rect.colliderect(lad.interact_rect)]
        self.on_ladder = False
        ladder_to_use = None
        # If continuing upwards without releasing UP, restrict to ladders with same X as last ladder
        if up_down and self.ladder_last_x is not None:
            ladder_candidates = [lad for lad in ladder_candidates if lad.x == self.ladder_last_x]
        if (up_down or down_down) and ladder_candidates:
            # If we were already on a ladder, stick to that ladder's x only
            if self.ladder_lock_x is not None:
                for lad in ladder_candidates:
                    if lad.x == self.ladder_lock_x:
                        ladder_to_use = lad
                        break
            # Otherwise start a climb on the closest ladder and lock to its x
            if ladder_to_use is None:
                ladder_to_use = min(ladder_candidates, key=lambda l: abs(l.x - (self.x + self.w//2)))
                self.ladder_lock_x = ladder_to_use.x

        if ladder_to_use is not None:
            self.on_ladder = True
            self.ladder_last_x = ladder_to_use.x
            # Center to the locked ladder only (no drifting to nearby ladders)
            # Snap exactly to ladder X
            self.x = ladder_to_use.x - self.w//2
            if keys[pygame.K_UP]:
                self.vy = -speed * 1.1
            elif keys[pygame.K_DOWN]:
                self.vy = speed * 1.1
            else:
                self.vy = 0.0
        else:
            # Not actively climbing; clear lock if we're not inside any locked-ladder area
            if not ladder_candidates:
                self.ladder_lock_x = None
            if not up_down:
                self.ladder_last_x = None
            self.vy += GRAVITY

        self.on_ground = False
        feet = pygame.Rect(self.rect.x, self.rect.bottom, self.w, 4)
        for p in sorted(platforms, key=lambda p: p.y):
            px1 = min(p.x1, p.x2)
            px2 = max(p.x1, p.x2)
            if feet.centerx >= px1 and feet.centerx <= px2:
                if abs(feet.top - p.y) <= 6 and self.vy >= 0:
                    self.on_ground = True
                    self.last_platform_y = p.y
                    break

        if self.rect.bottom >= HEIGHT and self.vy >= 0:
            self.on_ground = True
            self.last_platform_y = HEIGHT + 999

        if on_base_floor and self.on_ground:
            self.coyote = 0.15
        elif on_base_floor:
            self.coyote = max(0.0, self.coyote - dt)
        else:
            self.coyote = 0.0

        if jump_down and not self.jump_was_down and (self.on_ground or self.coyote > 0) and not self.on_ladder:
            self.vy = PLAYER_JUMP_VELOCITY
            if self.last_platform_y is not None and self.last_platform_y < HEIGHT:
                self.jumping = True
                self.jump_lock_y = self.last_platform_y
            else:
                self.jumping = True
                self.jump_lock_y = None
            self.coyote = 0.0

        self.x += self.vx
        self.y += self.vy

        # If climbing, allow passing through platforms ONLY when the next ladder segment above/below
        # is perfectly aligned (same x). Otherwise, stop on the platform.
        if self.on_ladder and self.ladder_lock_x is not None:
            feet = self.y + self.h
            aligned = [lad for lad in ladders if lad.x == self.ladder_lock_x]
            current = None
            for seg in aligned:
                if seg.y_top - 2 <= feet <= seg.y_bottom + 2:
                    current = seg
                    break
            if current is not None:
                if self.vy < 0 and feet < current.y_top - 1:
                    cont = None
                    for seg in aligned:
                        if seg.y_bottom == current.y_top:
                            cont = seg
                            break
                    if cont is None:
                        self.y = current.y_top - self.h
                        self.vy = 0.0
                        self.on_ladder = False
                        self.ladder_lock_x = None
                        self.last_platform_y = current.y_top
                        self.block_climb_until_up_release = True
                elif self.vy > 0 and feet > current.y_bottom + 1:
                    cont = None
                    for seg in aligned:
                        if seg.y_top == current.y_bottom:
                            cont = seg
                            break
                    if cont is None:
                        self.y = current.y_bottom - self.h
                        self.vy = 0.0
                        self.on_ladder = False
                        self.ladder_lock_x = None
                        self.last_platform_y = current.y_bottom
                        self.block_climb_until_up_release = True

        if self.x < 0: self.x = 0
        if self.x + self.w > WIDTH: self.x = WIDTH - self.w
        if self.y > HEIGHT - self.h:
            self.y = HEIGHT - self.h
            self.vy = 0.0

        if not self.on_ladder:
            prev_bottom = self.rect.bottom - self.vy
            for p in sorted(platforms, key=lambda p: p.y):
                if self.jumping and self.jump_lock_y is not None and p.y != self.jump_lock_y:
                    continue
                px1 = min(p.x1, p.x2)
                px2 = max(p.x1, p.x2)
                if self.rect.centerx >= px1 and self.rect.centerx <= px2:
                    if prev_bottom <= p.y <= self.rect.bottom and self.vy >= 0:
                        self.y = p.y - self.h
                        self.vy = 0.0
                        self.on_ground = True
                        self.last_platform_y = p.y
                        self.jumping = False
                        self.jump_lock_y = None
                        break

        if self.invuln_time > 0:
            self.invuln_time -= dt

        self.jump_was_down = jump_down
        self.up_was_down = up_down

    def draw(self, surf: pygame.Surface):
        if self.sprite_right is not None and self.sprite_left is not None:
            img = self.sprite_right if self.facing >= 0 else self.sprite_left
            if self.invuln_time > 0 and int(self.invuln_time * 15) % 2 == 0:
                alpha_prev = img.get_alpha()
                img.set_alpha(140)
                surf.blit(img, self.rect)
                img.set_alpha(alpha_prev if alpha_prev is not None else 255)
            else:
                surf.blit(img, self.rect)
            return

        c = self.color
        if self.invuln_time > 0 and int(self.invuln_time * 15) % 2 == 0:
            c = (180, 180, 180)
        pygame.draw.rect(surf, c, self.rect, border_radius=6)
        eye_y = self.rect.y + 12
        pygame.draw.circle(surf, WHITE, (self.rect.centerx - 6, eye_y), 3)
        pygame.draw.circle(surf, WHITE, (self.rect.centerx + 6, eye_y), 3)
        pygame.draw.circle(surf, BLACK, (self.rect.centerx - 6, eye_y), 1)
        pygame.draw.circle(surf, BLACK, (self.rect.centerx + 6, eye_y), 1)


class Barrel:
    def __init__(self, x: float, y: float, level_speed_mul: float, sticker: pygame.Surface):
        self.x = x
        self.y = y
        self.spawn_dir = random.choice([-1, 1])
        self.vx = self.spawn_dir * (BARREL_BASE_SPEED * level_speed_mul)
        self.vy = 0.0
        self.radius = BARREL_RADIUS
        self.spin = random.uniform(-5, 5)
        self.sticker = pygame.transform.smoothscale(sticker, (self.radius*2-8, self.radius*2-8))
        self.on_platform_index = None
        self.drop_cooldown = 0.3
        self.drop_target_index = None
        self.drop_target_y = None
        self._first_snap_done = False

    @property
    def rect(self) -> pygame.Rect:
        r = self.radius
        return pygame.Rect(int(self.x - r), int(self.y - r), r*2, r*2)

    def update(self, platforms: List[Platform], ladders: List[Ladder], dt: float):
        self.vy += GRAVITY
        self.drop_cooldown = max(0.0, self.drop_cooldown - dt)

        self.x += self.vx
        self.y += self.vy

        
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx)
        if self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -abs(self.vx)

        sorted_plats = sorted(platforms, key=lambda p: p.y)

        if self.on_platform_index is not None and 0 <= self.on_platform_index < len(sorted_plats):
            cur = sorted_plats[self.on_platform_index]
            cx1 = min(cur.x1, cur.x2); cx2 = max(cur.x1, cur.x2)
            leave_tol = 2
            catch_pad = 18
            if self.x < cx1 - leave_tol or self.x > cx2 + leave_tol:
                for j in range(self.on_platform_index + 1, len(sorted_plats)):
                    below = sorted_plats[j]
                    bx1 = min(below.x1, below.x2) - catch_pad
                    bx2 = max(below.x1, below.x2) + catch_pad
                    if bx1 <= self.x <= bx2:
                        self.drop_target_index = j
                        self.drop_target_y = below.y
                        self.x = min(max(self.x, bx1 + catch_pad), bx2 - catch_pad)
                        self.vx = 0.0
                        self.vy = max(self.vy, 1.2)
                        self.on_platform_index = None
                        self.drop_cooldown = max(0.15, self.drop_cooldown)
                        break

        prev_bottom = self.y - self.vy + self.radius

        snapped = False
        if self.drop_target_index is not None:
            tgt_i = self.drop_target_index
            if 0 <= tgt_i < len(sorted_plats):
                target = sorted_plats[tgt_i]
                plat_y = target.y
                if prev_bottom <= plat_y <= self.y + self.radius and self.vy >= 0:
                    self.y = plat_y - self.radius
                    self.vy = 0.0
                    self.on_platform_index = tgt_i
                    speed = max(1.2, abs(self.vx) or 1.2)
                    if not self._first_snap_done:
                        self.vx = speed * (self.spawn_dir or 1)
                        self._first_snap_done = True
                    else:
                        self.vx = math.copysign(speed, target.slope_dir)
                    self.drop_target_index = None
                    self.drop_target_y = None
                    snapped = True

        if not snapped:
            for i, p in enumerate(sorted_plats):
                px1 = min(p.x1, p.x2)
                px2 = max(p.x1, p.x2)
                if self.x >= px1 and self.x <= px2:
                    plat_y = p.y
                    if prev_bottom <= plat_y <= self.y + self.radius and self.vy >= 0:
                        self.y = plat_y - self.radius
                        self.vy = 0.0
                        self.on_platform_index = i
                        speed = max(1.2, abs(self.vx))
                        if not self._first_snap_done:
                            self.vx = speed * (self.spawn_dir or 1)
                            self._first_snap_done = True
                        else:
                            self.vx = math.copysign(speed, p.slope_dir)
                        break

        if self.on_platform_index is not None and self.drop_cooldown == 0.0:
            for lad in ladders:
                if self.rect.colliderect(lad.rect):
                    # 50% chance to drop through ladders
                    if random.random() < 0.5:
                        catch_pad = 18
                        # choose the first platform below whose x-range covers the ladder center
                        if 0 <= self.on_platform_index < len(sorted_plats):
                            for j in range(self.on_platform_index + 1, len(sorted_plats)):
                                below = sorted_plats[j]
                                bx1 = min(below.x1, below.x2) - catch_pad
                                bx2 = max(below.x1, below.x2) + catch_pad
                                if bx1 <= lad.x <= bx2:
                                    self.drop_target_index = j
                                    self.drop_target_y = below.y
                                    self.x = min(max(lad.x, bx1 + catch_pad), bx2 - catch_pad)
                                    self.vx = 0.0
                                    self.vy = max(self.vy, 1.2)
                                    self.on_platform_index = None
                                    self.drop_cooldown = 0.5
                                    break
                    break

    def draw(self, surf: pygame.Surface, t: float):
        r = self.radius
        cx, cy = int(self.x), int(self.y)

        pygame.draw.circle(surf, WOOD, (cx, cy), r)
        pygame.draw.circle(surf, BROWN, (cx, cy), r, 3)

        for k in (-8, 0, +8):
            pygame.draw.circle(surf, (100, 70, 30), (cx, cy), max(2, r + k), 2)

        rect = self.sticker.get_rect(center=(cx, cy))
        surf.blit(self.sticker, rect)


class Monkey:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.timer = 0.0

    def update(self, dt: float):
        self.timer += dt

    def draw(self, surf: pygame.Surface, t: float):
        body = pygame.Rect(self.x-24, self.y-28, 48, 56)
        pygame.draw.rect(surf, (100, 60, 30), body, border_radius=10)
        pygame.draw.circle(surf, (140, 90, 50), (body.centerx, body.top+18), 16)
        pygame.draw.circle(surf, BLACK, (body.centerx-5, body.top+15), 2)
        pygame.draw.circle(surf, BLACK, (body.centerx+5, body.top+15), 2)
        hand_y = body.centery + int(math.sin(t*4) * 6)
        pygame.draw.circle(surf, (100,60,30), (body.right+8, hand_y), 8)


# --- Simple particle for reward animation -------------------------------------------------------
class Particle:
    __slots__ = ("x","y","vx","vy","life")
    def __init__(self, x, y, vx, vy, life):
        self.x, self.y, self.vx, self.vy, self.life = x, y, vx, vy, life
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 60 * dt
        self.life -= dt
    def draw(self, surf):
        if self.life <= 0: return
        r = max(1, int(4 * self.life))
        pygame.draw.circle(surf, (240,240,240), (int(self.x), int(self.y)), r)


# --- UI: Buttons -------------------------------------------------------------------------------
@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str  # "start", "pause", "exit", "help", "scores", "sound"

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, hovered: bool=False, enabled: bool=True):
        base = (40, 40, 46)
        hover = (60, 60, 70)
        border = (120, 120, 130)
        fill = hover if (hovered and enabled) else base
        pygame.draw.rect(surf, fill, self.rect, border_radius=10)
        pygame.draw.rect(surf, border, self.rect, width=2, border_radius=10)
        color = (230,230,230) if enabled else (150,150,150)
        text = font.render(self.label, True, color)
        surf.blit(text, text.get_rect(center=self.rect.center))


# --- High scores helpers ------------------------------------------------------------------------
def load_scores() -> List[dict]:
    ensure_dirs()
    if not SAVE_PATH.exists():
        return []
    try:
        data = json.loads(SAVE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_scores(scores: List[dict]):
    ensure_dirs()
    try:
        SAVE_PATH.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    except Exception as e:
        pass



def qualifies_top_n(score: int, n: int = 10) -> bool:
    entries = load_scores()
    if len(entries) < n:
        return True
    try:
        entries_sorted = sorted(entries, key=lambda e: int(e.get("score", 0)), reverse=True)
    except Exception:
        entries_sorted = entries[:]
    if len(entries_sorted) < n:
        return True
    threshold = int(entries_sorted[n-1].get("score", 0))
    return int(score) >= threshold
def add_high_score(score: int, name: str = PLAYER_NAME):
    scores = load_scores()
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score": int(score),
        "name": str(name)
    }
    scores.append(entry)
    # keep only latest 100 by date
    save_scores(scores)


# --- Game ---------------------------------------------------------------------------------------
class Game:
    def __init__(self, debug: bool=False):
        pygame.init()
        pygame.display.set_caption("Kong (retro clone)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.bigfont = pygame.font.Font(FONT_NAME, 36)

        self.sfx = SFX(debug=debug)

        self.debug = debug

        self.player_sprite_surface = load_player_sprite(debug=self.debug)

        # Button bar setup
        self.buttons: List[Button] = []
        self._compute_buttons()

        self.reset_all()

    def _compute_buttons(self):
        labels = ["Start", "Pause", "Exit", "Help", "High Scores", "Sound on/off"]
        actions = ["start","pause","exit","help","scores","sound"]
        pad = 10
        btn_h = 34
        gap = 10
        # Make buttons smaller if needed to fit the width
        n = len(labels)
        avail = WIDTH - 2*pad
        btn_w = min(150, (avail - (n-1)*gap)//n)
        total_w = n*btn_w + (n-1)*gap
        start_x = max(10, (WIDTH - total_w)//2)
        y = HEIGHT - btn_h - pad
        rects = [pygame.Rect(start_x + i*(btn_w+gap), y, btn_w, btn_h) for i in range(n)]
        self.buttons = [Button(lbl, rect, act) for lbl, rect, act in zip(labels, rects, actions)]

    # ---------------- State & flow -------------
    # states: "intro", "playing", "reward_anim", "game_over"
    # overlay: None | "help" | "scores"
    def reset_all(self):
        self.level = 1
        self.score = 0
        self.lives = 3
        self.state = "intro"
        self.game_over = False
        self.intro_timer = 0.0
        self.reward_anim_type = None
        self.reward_timer = 0.0
        self.particles: List[Particle] = []
        self.confetti = []  # for 'shower' animation
        self.reward_extra = {}  # per-animation scratchpad
        self.overlay: Optional[str] = None
        self.paused = False
        self.was_playing_before_overlay = False
        self.gameover_logged = False
        self.name_input = PLAYER_NAME
        self.name_cursor = True
        self.cursor_timer = 0.0
        self.gameover_checked = False
        self.played_gameover = False
        self.new_level(show_intro=True)

    def new_level(self, show_intro: bool=True):
        self.platforms, self.ladders = build_level(self.level)
        start_plat = self.platforms[0]
        self.player = Player(40, start_plat.y - 60, sprite=self.player_sprite_surface)

        top_plat = self.platforms[-1]
        self.monkey = Monkey(top_plat.x1 + 30, top_plat.y - 20)

        self.reward_surf, self.reward_path = load_reward_logo_for_level(self.level)
        margin = 10
        gx = max(10, min(WIDTH - REWARD_W - margin, top_plat.x2 - REWARD_W - margin))
        gy = top_plat.y - REWARD_H - 12
        self.goal_rect = pygame.Rect(gx, gy, REWARD_W, REWARD_H)

        self.sticker_surface, self.sticker_saved_path = generate_level_thumbnail(self.level, size=44)

        self.barrels: List[Barrel] = []
        self.time_since_barrel = 0.0
        self.level_speed_mul = 1.0 + (self.level - 1) * 0.20
        self.next_barrel_interval = max(0.8, random.uniform(3.0, 4.0) / self.level_speed_mul)

        # Load transparent background overlay for this level
        self.bg_overlay, self.bg_path = load_random_background_overlay()

        if show_intro:
            self.state = "intro"
            self.intro_timer = 2.0
        else:
            self.state = "playing"

    def spawn_barrel(self):
        b = Barrel(self.monkey.x + 10, self.monkey.y + 10, self.level_speed_mul, self.sticker_surface)
        self.barrels.append(b)
    def handle_collisions(self, dt: float):
        for b in list(self.barrels):
            if self.player.rect.colliderect(b.rect):
                if self.player.invuln_time <= 0:
                    # Play hurt sound on valid hit
                    if hasattr(self, 'sfx'): self.sfx.play('hurt')
                    self.lives -= 1
                    self.player.invuln_time = 2.0
                    if self.lives < 0:
                        if not getattr(self, "played_gameover", False):
                            if hasattr(self, 'sfx'): self.sfx.play('game_over')
                            self.played_gameover = True
                        self.state = "game_over"

    def start_reward_animation(self):
        # Sound: reward
        if hasattr(self, 'sfx'): self.sfx.play('reward')

        # Choose among 7 animations
        anims = ["spin", "burst", "bounce", "flip", "zoomfade", "spiral", "shower"]
        self.reward_anim_type = random.choice(anims)
        # Default durations
        durations = {
            "spin": 1.4, "burst": 1.6, "bounce": 1.8, "flip": 1.6, "zoomfade": 1.5, "spiral": 1.7, "shower": 1.6
        }
        self.reward_timer = durations.get(self.reward_anim_type, 1.6)
        # Clear per-anim state
        self.particles.clear()
        self.confetti = []
        self.reward_extra = {}
        cx = self.goal_rect.centerx
        cy = self.goal_rect.centery

        if self.reward_anim_type == "burst":
            for i in range(60):
                ang = random.uniform(0, math.tau)
                speed = random.uniform(80, 220)
                vx = math.cos(ang) * speed
                vy = math.sin(ang) * speed
                life = random.uniform(0.6, 1.2)
                self.particles.append(Particle(cx, cy, vx, vy, life))

        elif self.reward_anim_type == "bounce":
            # Decaying vertical bounce
            self.reward_extra = {"amp": 140.0, "bounces": 3}

        elif self.reward_anim_type == "flip":
            # Simulated Y-flip using horizontal squash
            self.reward_extra = {"cycles": 2}

        elif self.reward_anim_type == "zoomfade":
            # No extra state required
            pass

        elif self.reward_anim_type == "spiral":
            # Spiral outward from center
            self.reward_extra = {"theta0": random.uniform(0, math.tau)}

        elif self.reward_anim_type == "shower":
            # Confetti rain around the reward
            colors = [(250,230,80), (120,200,250), (250,120,160), (160,250,180), (250,250,250)]
            for _ in range(90):
                x = cx + random.uniform(-180, 180)
                y = cy - random.uniform(120, 220)
                vx = random.uniform(-40, 40)
                vy = random.uniform(40, 120)
                life = random.uniform(0.8, 1.6)
                size = random.randint(2, 5)
                color = random.choice(colors)
                self.confetti.append({"x": x, "y": y, "vx": vx, "vy": vy, "life": life, "size": size, "color": color})
        self.state = "reward_anim"

    def update_reward_animation(self, dt: float):
        self.reward_timer -= dt
        if self.reward_anim_type == "burst":
            for p in list(self.particles):
                p.update(dt)
                if p.life <= 0:
                    self.particles.remove(p)
        elif self.reward_anim_type == "shower":
            # update confetti
            alive = []
            for c in self.confetti:
                c["x"] += c["vx"] * dt
                c["y"] += c["vy"] * dt
                c["vy"] += 240 * dt
                c["life"] -= dt
                if c["life"] > 0 and c["y"] < HEIGHT + 20:
                    alive.append(c)
            self.confetti = alive

        if self.reward_timer <= 0:
            self.level += 1
            self.score += 100
            # Prep next level state bits
            self.name_input = PLAYER_NAME
            self.name_cursor = True
            self.cursor_timer = 0.0
            self.gameover_checked = False
            self.played_gameover = False
            # Actually advance to next level
            self.new_level(show_intro=True)

    def open_overlay(self, name: str):
        if self.overlay == name:
            return
        self.was_playing_before_overlay = (self.state == "playing" and not self.paused)
        if self.was_playing_before_overlay:
            self.paused = True
        self.overlay = name

    def close_overlay(self):
        if self.overlay is not None:
            if self.was_playing_before_overlay:
                self.paused = False
            self.overlay = None
            self.was_playing_before_overlay = False

    def toggle_pause(self):
        if self.state == "reward_anim":
            return
        if self.state == "intro":
            # skip intro and start
            self.state = "playing"
            return
        if self.state == "game_over":
            return
        self.paused = not self.paused

    # --------------- Update/Drawing ---------------
    def update(self, dt: float):
        if self.state == "game_over":
            if not self.gameover_checked:
                if qualifies_top_n(self.score, 10):
                    self.open_overlay("enter_name")
                else:
                    add_high_score(self.score, PLAYER_NAME)
                self.gameover_checked = True
            return

        if self.overlay is not None:
            # UI overlay active: don't update world, but allow reward animation to continue if active
            return

        if self.state == "intro":
            self.intro_timer -= dt
            if self.intro_timer <= 0:
                self.state = "playing"
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
                self.state = "playing"
            return

        if self.state == "reward_anim":
            self.update_reward_animation(dt)
            return

        if self.paused:
            return

        keys = pygame.key.get_pressed()

        self.player.update(keys, self.platforms, self.ladders, dt)
        self.monkey.update(dt)

        for b in list(self.barrels):
            b.update(self.platforms, self.ladders, dt)
            if b.y - b.radius > HEIGHT + 80:
                self.barrels.remove(b)
                self.score += 5

        self.time_since_barrel += dt
        if self.time_since_barrel >= self.next_barrel_interval:
            self.time_since_barrel = 0.0
            self.spawn_barrel()
            self.next_barrel_interval = max(0.6, random.uniform(3.0, 4.0) / self.level_speed_mul)

        self.handle_collisions(dt)

        if self.player.rect.colliderect(self.goal_rect):
            self.player.vx = 0.0
            self.player.vy = 0.0
            self.start_reward_animation()

    def draw_hud(self):
        info = f"Level {self.level}   Score {self.score}   Lives {self.lives}"
        txt = self.font.render(info, True, WHITE)
        self.screen.blit(txt, (12, 8))

        # Debug diagnostics only if enabled
        if self.debug:
            if self.sticker_saved_path:
                s = str(self.sticker_saved_path)
                if len(s) > 58: s = "…" + s[-57:]
                small = self.font.render(f"Thumb: {s}", True, LIGHT)
                self.screen.blit(small, (12, 30))
            if PLAYER_SPRITE_PATH.exists():
                s2 = str(PLAYER_SPRITE_PATH)
                if len(s2) > 58: s2 = "…" + s2[-57:]
                small2 = self.font.render(f"Player: {s2}", True, LIGHT)
                self.screen.blit(small2, (12, 52))
            if self.reward_path is not None:
                s3 = str(self.reward_path)
                if len(s3) > 58: s3 = "…" + s3[-57:]
                small3 = self.font.render(f"Reward: {s3}", True, LIGHT)
                self.screen.blit(small3, (12, 74))

            if getattr(self, "bg_path", None) is not None:
                s4 = str(self.bg_path)
                if len(s4) > 58: s4 = "…"+s4[-57:]
                small4 = self.font.render(f"BG: {s4}", True, LIGHT)
                self.screen.blit(small4, (12, 96))

        # Button bar background strip
        strip_h = 48
        strip_rect = pygame.Rect(0, HEIGHT - strip_h, WIDTH, strip_h)
        pygame.draw.rect(self.screen, (18,18,24), strip_rect)
        pygame.draw.line(self.screen, (40,40,50), (0, HEIGHT - strip_h), (WIDTH, HEIGHT - strip_h), 2)
        # Button bar
        mx, my = pygame.mouse.get_pos()
        for btn in self.buttons:
            hovered = btn.rect.collidepoint(mx, my)
            # Dim the Sound button when muted (still clickable)
            if btn.action == "sound":
                enabled = not (hasattr(self, 'sfx') and getattr(self.sfx, 'muted', False))
            else:
                enabled = True
            btn.draw(self.screen, self.font, hovered=hovered, enabled=enabled)

    def draw_world(self, t: float):
        self.screen.fill((22, 22, 28))

        for p in self.platforms:
            rect = p.rect
            pygame.draw.rect(self.screen, (120, 50, 30), rect, border_radius=8)
            arrow_y = rect.y - 6
            direction = 1 if p.slope_dir > 0 else -1
            for x in range(rect.x + 20, rect.x + rect.w - 20, 60):
                pts = [(x, arrow_y), (x + 12*direction, arrow_y - 6), (x + 12*direction, arrow_y + 6)]
                pygame.draw.polygon(self.screen, (150, 80, 40), pts)

        for lad in self.ladders:
            r = lad.rect
            pygame.draw.rect(self.screen, (180, 170, 120), r, border_radius=6)
            for y in range(r.top+6, r.bottom-6, 14):
                pygame.draw.line(self.screen, (150, 130, 90), (r.left+4, y), (r.right-4, y), 3)

        if self.reward_surf is not None:
            shadow = pygame.Surface((REWARD_W, REWARD_H), pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0,0,0,120), pygame.Rect(0,0,REWARD_W,REWARD_H), border_radius=REWARD_RADIUS)
            self.screen.blit(shadow, (self.goal_rect.x+3, self.goal_rect.y+3))
            self.screen.blit(self.reward_surf, self.goal_rect)
            pygame.draw.rect(self.screen, (240,240,240), self.goal_rect, width=2, border_radius=REWARD_RADIUS)

        self.player.draw(self.screen)
        for b in self.barrels:
            b.draw(self.screen, t)

        self.monkey.draw(self.screen, t)

        self.draw_hud()

        # Background overlay (20% alpha) on top of world & HUD
        if getattr(self, "bg_overlay", None) is not None:
            self.screen.blit(self.bg_overlay, (0, 0))

        # Overlays
        if self.state == "intro":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,160))
            self.screen.blit(overlay, (0,0))
            title = self.bigfont.render(f"START LEVEL {self.level}", True, WHITE)
            sub = self.font.render("Press SPACE to start", True, LIGHT)
            self.screen.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2 - 10)))
            self.screen.blit(sub,   sub.get_rect(center=(WIDTH//2, HEIGHT//2 + 26)))

        elif self.state == "reward_anim":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,140))
            self.screen.blit(overlay, (0,0))

            cx, cy = self.goal_rect.center
            tleft = max(0.0, self.reward_timer)

            if self.reward_anim_type == "spin":
                norm = 1.0 - (tleft / 1.4)
                angle = norm * 360 * 2
                scale = 1.0 + 0.35 * math.sin(norm * math.tau * 2)
                img = pygame.transform.rotozoom(self.reward_surf, angle, scale)
                rect = img.get_rect(center=(cx, cy))
                self.screen.blit(img, rect)
                r = int(60 + 40 * norm)
                pygame.draw.circle(self.screen, (240,240,240), (cx, cy), r, 3)

            elif self.reward_anim_type == "burst":
                img = pygame.transform.rotozoom(self.reward_surf, 0, 1.1)
                rect = img.get_rect(center=(cx, cy))
                self.screen.blit(img, rect)
                for p in self.particles:
                    p.draw(self.screen)

            elif self.reward_anim_type == "bounce":
                dur = 1.8
                norm = 1.0 - (tleft / dur)
                amp0 = self.reward_extra.get("amp", 140.0)
                # 3 bounces with decay
                bounces = 3
                phase = norm * (bounces * math.pi)
                amp = amp0 * (1.0 - norm)  # decays to 0
                yoff = abs(math.sin(phase)) * amp
                # squash when near bottom
                factor = 1.0 - min(0.25, (yoff / max(1.0, amp0)) * 0.25)
                img = pygame.transform.rotozoom(self.reward_surf, 0, factor)
                rect = img.get_rect(center=(cx, cy - yoff + 10))
                # shadow
                shadow = pygame.Surface((REWARD_W, 16), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow, (0,0,0,150), shadow.get_rect())
                self.screen.blit(shadow, shadow.get_rect(center=(cx, cy + 26)))
                self.screen.blit(img, rect)

            elif self.reward_anim_type == "flip":
                dur = 1.6
                norm = 1.0 - (tleft / dur)
                cycles = self.reward_extra.get("cycles", 2)
                # scale X oscillates 1 -> 0 -> 1
                sx = max(0.06, abs(math.cos(norm * math.tau * cycles)))
                w = max(1, int(REWARD_W * sx))
                img = pygame.transform.smoothscale(self.reward_surf, (w, REWARD_H))
                rect = img.get_rect(center=(cx, cy))
                self.screen.blit(img, rect)
                # outline
                pygame.draw.rect(self.screen, (240,240,240), rect, 2, border_radius=REWARD_RADIUS)

            elif self.reward_anim_type == "zoomfade":
                dur = 1.5
                norm = 1.0 - (tleft / dur)
                s = 0.4 + 1.8 * norm
                img = pygame.transform.rotozoom(self.reward_surf, 0, s)
                img = img.convert_alpha()
                img.set_alpha(int(255 * (1.0 - norm)))
                rect = img.get_rect(center=(cx, cy))
                self.screen.blit(img, rect)

            elif self.reward_anim_type == "spiral":
                dur = 1.7
                norm = 1.0 - (tleft / dur)
                theta0 = self.reward_extra.get("theta0", 0.0)
                theta = theta0 + norm * math.tau * 4  # 2 spins
                radius = 20 + 180 * norm
                px = cx + int(math.cos(theta) * radius)
                py = cy + int(math.sin(theta) * radius)
                img = pygame.transform.rotozoom(self.reward_surf, norm * 360, 0.9 + 0.2*math.sin(norm*math.tau*2))
                rect = img.get_rect(center=(px, py))
                self.screen.blit(img, rect)
                # faint trail
                for k in range(1, 6):
                    f = max(0.0, 1.0 - k*0.18)
                    th = theta - k*0.35
                    rr = radius - k*14
                    tx = cx + int(math.cos(th) * rr)
                    ty = cy + int(math.sin(th) * rr)
                    pygame.draw.circle(self.screen, (200,200,220,50), (tx, ty), 6, 1)

            elif self.reward_anim_type == "shower":
                # draw static reward
                img = pygame.transform.rotozoom(self.reward_surf, 0, 1.0)
                rect = img.get_rect(center=(cx, cy))
                self.screen.blit(img, rect)
                # confetti
                for c in self.confetti:
                    s = c["size"]
                    r = pygame.Rect(int(c["x"]), int(c["y"]), s, s)
                    pygame.draw.rect(self.screen, c["color"], r)

        if self.paused and self.overlay is None and self.state == "playing":
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0,0,0,120))
            self.screen.blit(dim, (0,0))
            t1 = self.bigfont.render("PAUSED", True, WHITE)
            self.screen.blit(t1, t1.get_rect(center=(WIDTH//2, HEIGHT//2)))

        if self.state == "game_over":
            s1 = self.bigfont.render("GAME OVER", True, WHITE)
            s2 = self.font.render("Press F5 to restart, ESC to quit", True, LIGHT)
            self.screen.blit(s1, s1.get_rect(center=(WIDTH//2, HEIGHT//2 - 10)))
            self.screen.blit(s2, s2.get_rect(center=(WIDTH//2, HEIGHT//2 + 28)))

        # Help overlay
        if self.overlay == "help":
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0,0,0,180))
            self.screen.blit(dim, (0,0))
            box = pygame.Rect(0,0, min(700, WIDTH-120), min(420, HEIGHT-140))
            box.center = (WIDTH//2, HEIGHT//2)
            pygame.draw.rect(self.screen, (36,36,46), box, border_radius=16)
            pygame.draw.rect(self.screen, (160,160,170), box, width=2, border_radius=16)
            title = self.bigfont.render("Help", True, WHITE)
            self.screen.blit(title, title.get_rect(midtop=(box.centerx, box.top+18)))

            # Common text lines (no arrows yet)
            text_lines = [
                "Goal: climb to the top and collect the reward.",
                "Avoid barrels. Use ladders to move between platforms.",
                "",
                "Controls:"
            ]
            y = box.top + 70
            for line in text_lines:
                txt = self.font.render(line, True, LIGHT if line else LIGHT)
                self.screen.blit(txt, (box.left + 24, y))
                y += 28

            # --- Draw arrow keycaps so we don't rely on Unicode glyphs ---
            def draw_arrow_keycap(center, direction, size=34):
                w = size; h = size
                rect = pygame.Rect(0,0,w,h); rect.center = center
                pygame.draw.rect(self.screen, (26,26,34), rect, border_radius=8)
                pygame.draw.rect(self.screen, (150,150,160), rect, width=2, border_radius=8)
                cx, cy = rect.center
                s = int(size*0.30)
                if direction == "left":
                    pts = [(cx+s, cy-s), (cx-s, cy), (cx+s, cy+s)]
                elif direction == "right":
                    pts = [(cx-s, cy-s), (cx+s, cy), (cx-s, cy+s)]
                elif direction == "up":
                    pts = [(cx-s, cy+s), (cx, cy-s), (cx+s, cy+s)]
                else:  # down
                    pts = [(cx-s, cy-s), (cx, cy+s), (cx+s, cy-s)]
                pygame.draw.polygon(self.screen, (230,230,230), pts)

            # Row 1: ← →  move
            label = self.font.render("Arrow keys:", True, LIGHT)
            self.screen.blit(label, (box.left + 24, y))
            base_x = box.left + 24 + label.get_width() + 16
            cy = y + 12
            draw_arrow_keycap((base_x, cy), "left")
            draw_arrow_keycap((base_x + 46, cy), "right")
            lbl_move = self.font.render("move", True, LIGHT)
            self.screen.blit(lbl_move, (base_x + 46 + 40, y))
            y += 40

            # Row 2: ↑ ↓  climb
            spacer = self.font.render(" ", True, LIGHT)  # to keep alignment simple
            self.screen.blit(spacer, (box.left + 24, y))
            base_x2 = box.left + 24 + label.get_width() + 16
            cy2 = y + 12
            draw_arrow_keycap((base_x2, cy2), "up")
            draw_arrow_keycap((base_x2 + 46, cy2), "down")
            lbl_climb = self.font.render("climb", True, LIGHT)
            self.screen.blit(lbl_climb, (base_x2 + 46 + 40, y))
            y += 40

            # Other controls
            other = [
                "SPACE: jump",
                "F5: restart    ESC: quit",
                "F1 or Help button: open this help"
            ]
            for line in other:
                txt = self.font.render(line, True, LIGHT)
                self.screen.blit(txt, (box.left + 24, y))
                y += 28

            tip = self.font.render("Press ESC to close", True, WHITE)
            self.screen.blit(tip, tip.get_rect(midbottom=(box.centerx, box.bottom-14)))

        # Name entry overlay (on new Top 10)
        if self.overlay == "enter_name":
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0,0,0,190))
            self.screen.blit(dim, (0,0))
            box = pygame.Rect(0,0, min(720, WIDTH-120), 260)
            box.center = (WIDTH//2, HEIGHT//2)
            pygame.draw.rect(self.screen, (36,36,46), box, border_radius=16)
            pygame.draw.rect(self.screen, (160,160,170), box, width=2, border_radius=16)
            title = self.bigfont.render("New High Score!", True, WHITE)
            self.screen.blit(title, title.get_rect(midtop=(box.centerx, box.top+18)))
            prompt = self.font.render("Enter your name and press Enter to save. (ESC to cancel)", True, LIGHT)
            self.screen.blit(prompt, prompt.get_rect(midtop=(box.centerx, box.top+70)))
            # input box
            ibox = pygame.Rect(0,0, box.w - 120, 44)
            ibox.center = (box.centerx, box.centery + 10)
            pygame.draw.rect(self.screen, (26,26,34), ibox, border_radius=10)
            pygame.draw.rect(self.screen, (150,150,160), ibox, width=2, border_radius=10)
            # caret blink
            self.cursor_timer += 1/ FPS
            if self.cursor_timer >= 0.5:
                self.name_cursor = not self.name_cursor
                self.cursor_timer = 0.0
            display_name = self.name_input
            text_surface = self.bigfont.render(display_name, True, WHITE)
            text_rect = text_surface.get_rect(midleft=(ibox.left+14, ibox.centery))
            self.screen.blit(text_surface, text_rect)
            if self.name_cursor and (len(display_name) < 24):
                cx = text_rect.right + 6
                cy = ibox.centery
                pygame.draw.line(self.screen, WHITE, (cx, cy-12), (cx, cy+12), 2)
            tip = self.font.render("Saving will also open High Scores.", True, LIGHT)
            self.screen.blit(tip, tip.get_rect(midbottom=(box.centerx, box.bottom-14)))

        # High scores overlay
        if self.overlay == "scores":
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0,0,0,180))
            self.screen.blit(dim, (0,0))
            box = pygame.Rect(0,0, min(720, WIDTH-120), min(420, HEIGHT-120))
            box.center = (WIDTH//2, HEIGHT//2)
            pygame.draw.rect(self.screen, (36,36,46), box, border_radius=16)
            pygame.draw.rect(self.screen, (160,160,170), box, width=2, border_radius=16)
            title = self.bigfont.render("High Scores", True, WHITE)
            self.screen.blit(title, title.get_rect(midtop=(box.centerx, box.top+18)))
            entries = load_scores()
            # sort by score desc; show top 12
            entries_sorted = sorted(entries, key=lambda e: int(e.get("score",0)), reverse=True)[:12]
            header = self.font.render("Date                Score   Name", True, LIGHT)
            self.screen.blit(header, (box.left+24, box.top+70))
            y = box.top + 98
            if not entries_sorted:
                empty = self.font.render("No scores yet.", True, LIGHT)
                self.screen.blit(empty, (box.left+24, y))
            else:
                for e in entries_sorted:
                    date = str(e.get("date","")).ljust(18)[:18]
                    score = str(e.get("score",""))
                    name = str(e.get("name",""))
                    line = f"{date}   {score:>5}   {name}"
                    txt = self.font.render(line, True, WHITE)
                    self.screen.blit(txt, (box.left+24, y))
                    y += 26
            tip = self.font.render("Press ESC to close", True, WHITE)
            self.screen.blit(tip, tip.get_rect(midbottom=(box.centerx, box.bottom-14)))

    def _handle_button_click(self, pos):
        for btn in self.buttons:
            if btn.rect.collidepoint(pos):
                if btn.action == "start":
                    if self.state == "intro":
                        self.state = "playing"
                    elif self.state == "game_over":
                        self.reset_all()
                        self.state = "intro"
                    elif self.paused:
                        self.paused = False
                elif btn.action == "pause":
                    self.toggle_pause()
                elif btn.action == "exit":
                    self._external_quit = True
                elif btn.action == "help":
                    self.open_overlay("help")
                elif btn.action == "sound":
                    if hasattr(self, "sfx") and self.sfx:
                        self.sfx.toggle()
                elif btn.action == "scores":
                    self.open_overlay("scores")

    def run(self):
        self._external_quit = False
        title_timer = 0.0
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            title_timer += dt
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_button_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if self.overlay == "enter_name":
                        if event.key == pygame.K_ESCAPE:
                            self.close_overlay()
                        elif event.key == pygame.K_RETURN:
                            name = self.name_input.strip() or PLAYER_NAME
                            add_high_score(self.score, name)
                            self.close_overlay()
                            self.open_overlay("scores")
                        elif event.key == pygame.K_BACKSPACE:
                            self.name_input = self.name_input[:-1]
                        else:
                            ch = event.unicode
                            if ch and (len(self.name_input) < 18) and (32 <= ord(ch) < 127):
                                # Allow simple printable ASCII
                                self.name_input += ch
                        continue
                    if event.key == pygame.K_ESCAPE:
                        if self.overlay is not None:
                            self.close_overlay()
                        else:
                            running = False
                    elif event.key == pygame.K_F5:
                        self.reset_all()
                    elif event.key == pygame.K_F1:
                        if self.overlay == "help":
                            self.close_overlay()
                        else:
                            self.open_overlay("help")

            if self._external_quit:
                running = False

            self.update(dt)
            self.draw_world(title_timer)
            pygame.display.flip()

        pygame.quit()


def main():
    ensure_dirs()

    debug = ('--debug' in sys.argv) or (os.getenv('KONG_DEBUG','').strip() in ('1','true','True'))
    if debug:
        print(f"[Kong] ROOT: {ROOT}")
        print(f"[Kong] Startup dir (source JPGs): {STARTUP_DIR}")
        print(f"[Kong] Thumbs dir (circular PNGs): {THUMBS_DIR}")
        print(f"[Kong] Assets dir: {ASSETS_DIR}")
        print(f"[Kong] Custom player path: {PLAYER_SPRITE_PATH}")
        print(f"[Kong] High score file: {SAVE_PATH}")
        if not STARTUP_DIR.exists():
            print("[Kong] WARNING: Startup dir does not exist. Place logo_1.jpg etc. for reward, or any JPGs for barrel stickers.")
        else:
            try:
                jpgs = [p.name for p in STARTUP_DIR.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')]
                logos = [p.name for p in STARTUP_DIR.iterdir() if p.name.lower().startswith('logo_') and p.suffix.lower() in ('.jpg', '.jpeg')]
                print(f"[Kong] Found {len(jpgs)} JPG(s) total; {len(logos)} logo_* file(s)." )
            except Exception as e:
                print(f"[Kong] Could not enumerate JPGs in {STARTUP_DIR}: {e}")

        if PLAYER_SPRITE_PATH.exists():
            print("[Kong] Custom player sprite detected. It will be used in-game.")
        else:
            print("[Kong] No custom player sprite detected; using default box player.")

    game = Game(debug=debug)
    game.run()


if __name__ == "__main__":
    main()
