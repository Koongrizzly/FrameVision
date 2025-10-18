#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donkey Kong–style mini-clone built with Pygame.

- File location: /helpers/kong.py
- Thumbnails: each level, the game picks a random JPG from ROOT/presets/startup/,
  makes a circular PNG "sticker", and saves it to ROOT/presets/setsave/thumbs/kong/ .
- Those rounded thumbnails are then rendered inside the barrels for that level.
- Simple platforms/ladders, barrels roll with alternating slopes and occasionally drop down ladders.
- Controls: ← → to move, ↑/↓ to climb, SPACE to jump, ESC to quit, F5 to reset.
- To run: `python helpers/kong.py` (from the project root), or `python -m helpers.kong`

This is a lightweight educational clone, not a 1:1 recreation. No external assets required.
"""

import os
import sys
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# --- Pygame bootstrap ---------------------------------------------------------------------------
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


# --- Paths (relative to this file so it works inside /helpers/) ---------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # go from /helpers/kong.py → project ROOT
STARTUP_DIR = ROOT / "presets" / "startup"
THUMBS_DIR  = ROOT / "presets" / "setsave" / "thumbs" / "kong"


# --- Utils --------------------------------------------------------------------------------------
def ensure_dirs():
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def pick_random_jpg(source_dir: Path) -> Optional[Path]:
    if not source_dir.exists():
        return None
    exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
    jpgs = [p for p in source_dir.iterdir() if p.suffix in exts and p.is_file()]
    if not jpgs:
        return None
    return random.choice(jpgs)


def make_circular_thumbnail_pygame(src_path: Path, size: int) -> pygame.Surface:
    """
    Pure Pygame approach: load image, scale square, apply circular alpha mask via BLEND_RGBA_MULT.
    Returns a Pygame Surface with per-pixel alpha (circle visible, outside transparent).
    """
    img = pygame.image.load(str(src_path)).convert_alpha()
    # Make it square (crop center) then scale
    w, h = img.get_width(), img.get_height()
    side = min(w, h)
    # crop rect
    crop_rect = pygame.Rect((w - side)//2, (h - side)//2, side, side)
    square = pygame.Surface((side, side), pygame.SRCALPHA)
    square.blit(img, (0, 0), crop_rect)

    # scale
    thumb = pygame.transform.smoothscale(square, (size, size)).convert_alpha()

    # create circular mask
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (size//2, size//2), size//2)
    # apply mask (multiply RGBA; outside circle becomes transparent)
    thumb.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return thumb


def save_surface_png(surface: pygame.Surface, path: Path):
    # Ensure parent exists
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def generate_level_thumbnail(level_index: int, size: int = 44) -> Tuple[pygame.Surface, Optional[Path]]:
    """
    Each level: pick a random JPG from STARTUP_DIR, generate a circular PNG,
    save to THUMBS_DIR as 'thumb_level_XXX.png', and return the Pygame surface + saved path.
    If no source JPGs exist, generate a placeholder graphic.
    """
    ensure_dirs()
    src = pick_random_jpg(STARTUP_DIR)
    out_path = THUMBS_DIR / f"thumb_level_{level_index:03d}.png"

    if src is None:
        # Create placeholder (question mark in a circle)
        size = max(32, size)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (220, 220, 220, 255), (size//2, size//2), size//2)
        pygame.draw.circle(surf, (90, 90, 90, 255), (size//2, size//2), size//2, 3)
        font = pygame.font.Font(FONT_NAME, max(14, size//2))
        q = font.render("?", True, (90, 90, 90))
        q_rect = q.get_rect(center=(size//2, size//2))
        surf.blit(q, q_rect)
        # Save placeholder thumb
        save_surface_png(surf, out_path)
        return surf, out_path

    # Build thumb (pure pygame path so no Pillow dependency)
    try:
        surf = make_circular_thumbnail_pygame(src, size=size)
        save_surface_png(surf, out_path)
        return surf, out_path
    except Exception as e:
        # Fallback: placeholder on error
        size = max(32, size)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (220, 220, 220, 255), (size//2, size//2), size//2)
        pygame.draw.circle(surf, (200, 50, 50, 255), (size//2, size//2), size//2, 3)
        font = pygame.font.Font(FONT_NAME, max(12, size//3))
        t = font.render("ERR", True, (200, 50, 50))
        t_rect = t.get_rect(center=(size//2, size//2))
        surf.blit(t, t_rect)
        save_surface_png(surf, out_path)
        return surf, out_path


# --- Level geometry -----------------------------------------------------------------------------
@dataclass
class Platform:
    x1: int
    x2: int
    y: int
    slope_dir: int  # -1 = slopes left (barrels roll left), +1 = slopes right

    @property
    def rect(self) -> pygame.Rect:
        # a thin platform strip (used for drawing; collision is handled logically)
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
        # Slightly extend above and below platforms so the player can 'catch' the ladder
        pad = 24
        return pygame.Rect(self.x - 12, self.y_top - pad, 24, (self.y_bottom - self.y_top) + pad*2)




def build_level(level: int) -> Tuple[List[Platform], List[Ladder]]:
    """
    Strong catch solver:
    - Build TOP→BOTTOM with alternating slopes.
    - When previous row drops RIGHT, the new row is short & right-anchored and
      its [x1,x2] is chosen so prev.x2 lies at least CATCH_PAD inside it.
    - When previous row drops LEFT, the new row is full & left-anchored; with GUARD.
    """
    rng = random.Random(5050 + level * 111)
    margin = 60
    GUARD = 40          # pull-in for droppers so they don't touch the wall
    CATCH_PAD = 100     # how far inside the catcher the drop x must land
    gap = 100
    plat_count = 5

    platforms: List[Platform] = []
    ladders: List[Ladder] = []

    left_wall, right_wall = margin, WIDTH - margin
    full_w = right_wall - left_wall
    short_w = max(520, int(full_w * 0.60))  # narrower to emphasize zig-zag

    top_y0 = HEIGHT - 80 - (plat_count - 1) * gap

    for idx in range(plat_count):
        y = top_y0 + idx * gap
        if idx == 0:
            # Top gameplay row: full-left anchored, drop to RIGHT (inside by GUARD)
            x1 = left_wall
            x2 = right_wall - GUARD
            slope = +1
        else:
            prev = platforms[-1]
            if prev.slope_dir == +1:
                # Must catch RIGHT edge (prev.x2)
                x2 = right_wall
                effective_w = max(short_w, CATCH_PAD*2 + 200)
                x1_max = x2 - effective_w
                target_x1 = int(prev.x2 - (effective_w - CATCH_PAD))
                x1 = max(left_wall + GUARD, min(x1_max, target_x1))
                x2 = x1 + effective_w
                slope = -1
            else:
                # Must catch LEFT edge (prev.x1)
                x1 = left_wall
                x2 = right_wall - GUARD
                slope = +1

        # tiny jitter preserving constraints
        if slope == +1:
            off = rng.randint(-6, 6)
            x1 = max(left_wall, x1 + off)
            x2 = min(right_wall - GUARD, x2 + off)
        else:
            off = rng.randint(-10, 10)
            x1 = max(left_wall + GUARD, x1 + off)
            x2 = min(right_wall, x2 + off)

        platforms.append(Platform(int(x1), int(x2), int(y), slope))

    # Top flat ledge
    top_w = int(WIDTH * 0.60)
    top_y = max(90, top_y0 - 80)
    top_x1 = (WIDTH - top_w) // 2
    platforms.append(Platform(top_x1, top_x1 + top_w, top_y, +1))

    # Ladders
    for i in range(len(platforms) - 1):
        lower = platforms[i]     # larger y
        upper = platforms[i+1]   # smaller y
        left = max(margin, min(lower.x1, lower.x2))
        right = min(WIDTH - margin, max(lower.x1, lower.x2))

        if level <= 3:
            xs = [
                int(left + (right - left) * 1/3 + rng.randint(-24, 24)),
                int(left + (right - left) * 2/3 + rng.randint(-24, 24)),
            ]
        else:
            cnt = rng.randint(1, 3)
            xs = [rng.randint(left + 40, right - 40) for _ in range(cnt)]
        for x in xs:
            ladders.append(Ladder(int(x), upper.y, lower.y))

    return platforms, ladders
# --- Entities -----------------------------------------------------------------------------------
class Player:
    def __init__(self, x: float, y: float):
        self.w = 28
        self.h = 38
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.on_ladder = False
        self.invuln_time = 0.0  # brief invulnerability after hit
        self.coyote = getattr(self, 'coyote', 0.0)
        self.last_platform_y: Optional[int] = None
        self.jumping: bool = False
        self.jump_lock_y: Optional[int] = None
        self.jump_was_down: bool = False
        self.coyote = 0.0  # small grace period to still allow jumping after stepping off ledge
        self.color = BLUE

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def update(self, keys, platforms: List[Platform], ladders: List[Ladder], dt: float):
        speed = PLAYER_SPEED
        self.vx = 0.0
        jump_down = keys[pygame.K_SPACE]

        # Ladder test
        overlapping_ladder = None
        for lad in ladders:
            # Use a padded interaction rect so you can start climbing from the platform
            if self.rect.colliderect(lad.interact_rect):
                overlapping_ladder = lad
                break
        # Are we on the base floor (screen bottom)?
        on_base_floor = (self.rect.bottom >= HEIGHT - 1)

        # Movement
        if keys[pygame.K_LEFT]:
            self.vx = -speed
        if keys[pygame.K_RIGHT]:
            self.vx = speed

        # Ladder movement
        self.on_ladder = False
        if overlapping_ladder and (keys[pygame.K_UP] or keys[pygame.K_DOWN]):
            self.on_ladder = True
            # Snap x near ladder center to make climbing feel nicer
            offset = (overlapping_ladder.x - (self.x + self.w//2))
            self.x += offset * 0.6
            if abs(offset) < 2:
                self.x = overlapping_ladder.x - self.w//2
            if keys[pygame.K_UP]:
                self.vy = -speed * 1.1
            elif keys[pygame.K_DOWN]:
                self.vy = speed * 1.1
            else:
                self.vy = 0.0
        else:
            # gravity when not climbing
            self.vy += GRAVITY

        # Jump (only if on ground and not on ladder)
        # set on_ground by checking platforms below
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
        # Consider screen bottom as ground so you can jump from the floor
        if self.rect.bottom >= HEIGHT and self.vy >= 0:
            self.on_ground = True
            self.last_platform_y = HEIGHT + 999  # sentinel for base floor
        # Update coyote time (jump forgiveness) ONLY on base floor
        if on_base_floor and self.on_ground:
            self.coyote = 0.15
        elif on_base_floor:
            self.coyote = max(0.0, self.coyote - dt)
        else:
            self.coyote = 0.0

        # Space to jump
        if jump_down and not self.jump_was_down and (self.on_ground or self.coyote > 0) and not self.on_ladder:
            self.vy = PLAYER_JUMP_VELOCITY
            # Start of a jump: lock to current floor unless on base floor
            if self.last_platform_y is not None and self.last_platform_y < HEIGHT:
                self.jumping = True
                self.jump_lock_y = self.last_platform_y
            else:
                # From base floor, allow landing on the first platform
                self.jumping = True
                self.jump_lock_y = None
            self.coyote = 0.0
            self.coyote = 0.0

        # Integrate
        self.x += self.vx
        self.y += self.vy

        # World bounds
        if self.x < 0: self.x = 0
        if self.x + self.w > WIDTH: self.x = WIDTH - self.w
        if self.y > HEIGHT - self.h:
            self.y = HEIGHT - self.h
            self.vy = 0.0

        # Land on platforms (snap when crossing downward)
        if not self.on_ladder:
            prev_bottom = self.rect.bottom - self.vy
            for p in sorted(platforms, key=lambda p: p.y):
                # If we're in a jump with a lock, only allow landing back on the locked floor
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
                        # Landing ends the jump
                        self.jumping = False
                        self.jump_lock_y = None
                        break

        # Invulnerability timer
        if self.invuln_time > 0:
            self.invuln_time -= dt

        # Remember jump key state to require release before next jump
        self.jump_was_down = jump_down

    def draw(self, surf: pygame.Surface):
        c = self.color
        if self.invuln_time > 0 and int(self.invuln_time * 15) % 2 == 0:
            c = (180, 180, 180)
        pygame.draw.rect(surf, c, self.rect, border_radius=6)
        # face
        eye_y = self.rect.y + 12
        pygame.draw.circle(surf, WHITE, (self.rect.centerx - 6, eye_y), 3)
        pygame.draw.circle(surf, WHITE, (self.rect.centerx + 6, eye_y), 3)
        pygame.draw.circle(surf, BLACK, (self.rect.centerx - 6, eye_y), 1)
        pygame.draw.circle(surf, BLACK, (self.rect.centerx + 6, eye_y), 1)


class Barrel:
    def __init__(self, x: float, y: float, level_speed_mul: float, sticker: pygame.Surface):
        self.x = x
        self.y = y
        self.vx = random.choice([-1, 1]) * (BARREL_BASE_SPEED * level_speed_mul)
        self.vy = 0.0
        self.radius = BARREL_RADIUS
        self.spin = random.uniform(-5, 5)
        self.sticker = pygame.transform.smoothscale(sticker, (self.radius*2-8, self.radius*2-8))
        self.on_platform_index = None  # which platform we're rolling on (if any)
        self.drop_cooldown = 0.3  # short cooldown before we allow ladder dropping

    @property
    def rect(self) -> pygame.Rect:
        r = self.radius
        return pygame.Rect(int(self.x - r), int(self.y - r), r*2, r*2)

    def update(self, platforms: List[Platform], ladders: List[Ladder], dt: float):
        self.vy += GRAVITY
        self.drop_cooldown = max(0.0, self.drop_cooldown - dt)

        # Integrate motion
        self.x += self.vx
        self.y += self.vy

        # World bounce on walls a bit (keep barrels on screen)
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx)
        if self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -abs(self.vx)

        # Collide with platforms from above
        prev_bottom = self.y - self.vy + self.radius
        for i, p in enumerate(sorted(platforms, key=lambda p: p.y)):
            px1 = min(p.x1, p.x2)
            px2 = max(p.x1, p.x2)
            if self.x >= px1 and self.x <= px2:
                plat_y = p.y
                if prev_bottom <= plat_y <= self.y + self.radius and self.vy >= 0:
                    # Land on platform
                    self.y = plat_y - self.radius
                    self.vy = 0.0
                    self.on_platform_index = i
                    # roll along slope
                    speed = max(1.2, abs(self.vx))
                    self.vx = math.copysign(speed, p.slope_dir)
                    break

        # Possibly drop down a ladder
        if self.on_platform_index is not None and self.drop_cooldown == 0.0:
            for lad in ladders:
                if self.rect.colliderect(lad.rect):
                    if random.random() < 0.15:  # 15% chance to drop
                        self.x = lad.x
                        self.vx = 0.0
                        self.vy = 1.0
                        self.on_platform_index = None
                        self.drop_cooldown = 0.5
                        break

    def draw(self, surf: pygame.Surface, t: float):
        # Draw a cartoony barrel: body + hoops
        r = self.radius
        cx, cy = int(self.x), int(self.y)

        # body
        pygame.draw.circle(surf, WOOD, (cx, cy), r)
        pygame.draw.circle(surf, BROWN, (cx, cy), r, 3)

        # hoops (animate via time)
        phase = int((t * 6) % 360)
        for k in (-8, 0, +8):
            pygame.draw.circle(surf, (100, 70, 30), (cx, cy), max(2, r + k), 2)

        # sticker (thumbnail) in the center
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
        # Big silly monkey
        body = pygame.Rect(self.x-24, self.y-28, 48, 56)
        pygame.draw.rect(surf, (100, 60, 30), body, border_radius=10)
        pygame.draw.circle(surf, (140, 90, 50), (body.centerx, body.top+18), 16)
        pygame.draw.circle(surf, BLACK, (body.centerx-5, body.top+15), 2)
        pygame.draw.circle(surf, BLACK, (body.centerx+5, body.top+15), 2)
        # waving hand
        hand_y = body.centery + int(math.sin(t*4) * 6)
        pygame.draw.circle(surf, (100,60,30), (body.right+8, hand_y), 8)


# --- Game ---------------------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Kong (retro clone)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.bigfont = pygame.font.Font(FONT_NAME, 32)

        self.reset_all()

    def reset_all(self):
        self.level = 1
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.new_level()

    def new_level(self):
        self.platforms, self.ladders = build_level(self.level)
        # Player starts bottom-left
        start_plat = self.platforms[0]
        self.player = Player(40, start_plat.y - 38)

        # Monkey sits on top-left of top platform
        top_plat = self.platforms[-1]
        self.monkey = Monkey(top_plat.x1 + 30, top_plat.y - 20)

        # Goal position (top-right)
        self.goal_rect = pygame.Rect(top_plat.x2 - 40, top_plat.y - 45, 32, 32)

        # Sticker for barrels (generated & saved)
        self.sticker_surface, self.sticker_saved_path = generate_level_thumbnail(self.level, size=44)

        # Barrels
        self.barrels: List[Barrel] = []
        self.time_since_barrel = 0.0
        self.level_speed_mul = 1.0 + (self.level - 1) * 0.20  # slightly faster each level
        # Next barrel timer (random between 3–4s at level 1, faster on later levels)
        self.next_barrel_interval = max(0.8, random.uniform(3.0, 4.0) / self.level_speed_mul)

    def spawn_barrel(self):
        b = Barrel(self.monkey.x + 10, self.monkey.y + 10, self.level_speed_mul, self.sticker_surface)
        self.barrels.append(b)

    def handle_collisions(self, dt: float):
        # Player vs barrels
        for b in list(self.barrels):
            if self.player.rect.colliderect(b.rect):
                if self.player.invuln_time <= 0:
                    self.lives -= 1
                    self.player.invuln_time = 2.0
                    if self.lives < 0:
                        self.game_over = True

    def update(self, dt: float):
        if self.game_over:
            return

        keys = pygame.key.get_pressed()
        self.player.update(keys, self.platforms, self.ladders, dt)
        self.monkey.update(dt)

        for b in list(self.barrels):
            b.update(self.platforms, self.ladders, dt)
            # remove off-screen bottoms
            if b.y - b.radius > HEIGHT + 80:
                self.barrels.remove(b)
                self.score += 5

        # spawn barrels
        self.time_since_barrel += dt
        if self.time_since_barrel >= self.next_barrel_interval:
            self.time_since_barrel = 0.0
            self.spawn_barrel()
            # schedule the next one
            self.next_barrel_interval = max(0.6, random.uniform(3.0, 4.0) / self.level_speed_mul)

        self.handle_collisions(dt)

        # Check goal reach
        if self.player.rect.colliderect(self.goal_rect):
            self.score += 100
            self.level += 1
            self.new_level()

    def draw_hud(self):
        info = f"Level {self.level}   Score {self.score}   Lives {self.lives}"
        txt = self.font.render(info, True, WHITE)
        self.screen.blit(txt, (12, 8))

        # Show path of latest saved thumbnail (shortened) to confirm I/O
        if self.sticker_saved_path:
            s = str(self.sticker_saved_path)
            if len(s) > 58:
                s = "…" + s[-57:]
            small = self.font.render(f"Thumb: {s}", True, LIGHT)
            self.screen.blit(small, (12, 30))

        # Controls
        ctrl = self.font.render("←/→ move   ↑/↓ climb   SPACE jump   F5 reset   ESC quit", True, LIGHT)
        self.screen.blit(ctrl, (12, HEIGHT-26))

    def draw_world(self, t: float):
        self.screen.fill((22, 22, 28))

        # platforms
        for p in self.platforms:
            rect = p.rect
            pygame.draw.rect(self.screen, (120, 50, 30), rect, border_radius=8)
            # slope hint arrows
            arrow_y = rect.y - 6
            direction = 1 if p.slope_dir > 0 else -1
            for x in range(rect.x + 20, rect.x + rect.w - 20, 60):
                pts = [(x, arrow_y), (x + 12*direction, arrow_y - 6), (x + 12*direction, arrow_y + 6)]
                pygame.draw.polygon(self.screen, (150, 80, 40), pts)

        # ladders
        for lad in self.ladders:
            r = lad.rect
            pygame.draw.rect(self.screen, (180, 170, 120), r, border_radius=6)
            # rungs
            for y in range(r.top+6, r.bottom-6, 14):
                pygame.draw.line(self.screen, (150, 130, 90), (r.left+4, y), (r.right-4, y), 3)

        # goal (princess/flag)
        pygame.draw.rect(self.screen, (230, 210, 230), self.goal_rect, border_radius=8)
        pygame.draw.circle(self.screen, RED, self.goal_rect.center, 6)

        # entities
        self.player.draw(self.screen)
        for b in self.barrels:
            b.draw(self.screen, t)

        self.monkey.draw(self.screen, t)

        self.draw_hud()

    def run(self):
        # Title flash
        title_timer = 0.0
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            title_timer += dt
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_F5:
                        self.reset_all()

            self.update(dt)
            self.draw_world(title_timer)

            # Game over overlay
            if self.game_over:
                s1 = self.bigfont.render("GAME OVER", True, WHITE)
                s2 = self.font.render("Press F5 to restart, ESC to quit", True, LIGHT)
                self.screen.blit(s1, s1.get_rect(center=(WIDTH//2, HEIGHT//2 - 10)))
                self.screen.blit(s2, s2.get_rect(center=(WIDTH//2, HEIGHT//2 + 28)))

            pygame.display.flip()

        pygame.quit()


def main():
    # If launched directly, run the game
    # Ensure directories exist early so first-level thumb can be saved
    ensure_dirs()

    # Quick console message about paths
    print(f"[Kong] ROOT: {ROOT}")
    print(f"[Kong] Startup dir (source JPGs): {STARTUP_DIR}")
    print(f"[Kong] Thumbs dir (circular PNGs): {THUMBS_DIR}")
    if not STARTUP_DIR.exists():
        print("[Kong] WARNING: Startup dir does not exist. A placeholder thumbnail will be used.")
    else:
        # List a few found JPGs
        try:
            jpgs = [p.name for p in STARTUP_DIR.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')]
            print(f"[Kong] Found {len(jpgs)} JPG(s).")
        except Exception as e:
            print(f"[Kong] Could not enumerate JPGs in {STARTUP_DIR}: {e}")

    game = Game()
    game.run()


if __name__ == "__main__":
    main()
