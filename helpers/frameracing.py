#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrameRacing — retro 4-lane racer (Pygame)

Paths
- Game: root/helpers/frameracing.py
- High scores: root/setsave/presets/racescore.json
- Backgrounds: root/presets/startup/  (15% opacity overlay, changes on level-up)
- Car sprites: root/assets/cars/  (player.png, enemy_*.png, special_*.png)

Notes
- Normal pass: +20 points; Special pass: +50 points
- Special cars are faster and can be animated (multiple special_*.png frames)
- Cars >50% off-screen at bottom become harmless
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)

import pygame

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[1]
HELPERS_DIR = ROOT_DIR / "helpers"
SCORES_PATH = ROOT_DIR / "setsave" / "presets" / "racescore.json"
BG_DIR = ROOT_DIR / "presets" / "startup"
ASSETS_DIR = ROOT_DIR / "assets" / "cars"

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
START_MAXIMIZED = True   # windowed near-max (not fullscreen)
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

WIDTH, HEIGHT = 980, 900
FPS = 60
PLAYER_VERT_SPEED = 420.0  # pixels/sec when holding ↑/↓
LANES = 4

# Panels and road
PANEL_LEFT_W = 200
PANEL_RIGHT_W = 260
ROAD_MARGIN = 36

ROAD_COLOR = (22, 22, 26)
LANE_COLOR = (64, 64, 72)

PLAYER_W, PLAYER_H = 56, 98
ENEMY_W, ENEMY_H = 56, 98
PU_SIZE = 44

# Level & spawn
START_REQUIRED = 8
REQUIRED_INC = 6
BASE_ENEMY_SPEED = 160.0        # pixels/sec
LEVEL_SPEED_INC = 18.0          # px/s per level
SPAWN_MS_START = 900
SPAWN_MS_MIN = 380
SPAWN_DEC_PER_LEVEL = 60

# Power-ups
POWERUP_SPAWN_CHANCE = 0.12
POWERUP_DURATION = {
    "FAST": 7.0,
    "SLOW": 5.0,
    "SHIELD": 12.0,
    "SHOOT": 8.0,
}
SHOOT_COOLDOWN = 0.5  # seconds

# Scoring & special enemies
NORMAL_PASS_POINTS = 20
SPECIAL_PASS_POINTS = 50
SPECIAL_ENEMY_CHANCE = 0.18
SPECIAL_SPEED_FACTOR = 1.35

# Background
BG_ALPHA = int(255 * 0.35)  # 55% opacity

# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------
pygame.init()
pygame.display.set_caption("FrameRacing")

def get_desktop_size():
    try:
        sizes = pygame.display.get_desktop_sizes()
        if sizes:
            return sizes[0]
    except Exception:
        pass
    info = pygame.display.Info()
    return info.current_w, info.current_h

if START_MAXIMIZED:
    dw, dh = get_desktop_size()
    WIDTH = max(800, dw - 80)
    HEIGHT = max(600, dh - 120)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()

# Fonts
FONT = pygame.font.SysFont("consolas", 22)
FONT_BIG = pygame.font.SysFont("consolas", 36, bold=True)
FONT_HUGE = pygame.font.SysFont("consolas", 64, bold=True)
PANEL_FONT = pygame.font.SysFont("consolas", 18)
PANEL_TITLE_FONT = pygame.font.SysFont("consolas", 24, bold=True)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def road_rect():
    return pygame.Rect(PANEL_LEFT_W, 0, WIDTH - PANEL_LEFT_W - PANEL_RIGHT_W, HEIGHT)

def lanes_geometry():
    rr = road_rect()
    road_w = rr.width - 2 * ROAD_MARGIN
    lane_w = max(40, road_w / max(1, LANES))
    centers = [int(rr.left + ROAD_MARGIN + lane_w * (i + 0.5)) for i in range(LANES)]
    rects = [pygame.Rect(int(rr.left + ROAD_MARGIN + i * lane_w), 0, int(lane_w), HEIGHT) for i in range(LANES)]
    return centers, rects, int(lane_w)

LANE_CENTERS, LANE_RECTS, LANE_W = lanes_geometry()

def recalc_geometry(player=None):
    global LANE_CENTERS, LANE_RECTS, LANE_W
    LANE_CENTERS, LANE_RECTS, LANE_W = lanes_geometry()
    if player is not None:
        # Keep X centered to lane; preserve Y within road bounds.
        player.center_to_lane()
        rr = road_rect()
        player.rect.y = clamp(player.rect.y, rr.top + 6, rr.bottom - player.rect.height - 6)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def draw_text(surf, text, pos, font=FONT, color=(240, 240, 240), center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surf.blit(img, rect)
    return rect

def ensure_scores_file():
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SCORES_PATH.exists():
        SCORES_PATH.write_text("[]", encoding="utf-8")

def load_scores():
    ensure_scores_file()
    try:
        with open(SCORES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def save_score(name, score):
    scores = load_scores()
    scores.append({"name": name[:16] or "PLAYER", "score": int(score), "date": datetime.now().isoformat(timespec="seconds")})
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:10]
    try:
        with open(SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return scores

# Background helpers
def list_bg_images():
    if not BG_DIR.exists():
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        files += list(BG_DIR.glob(ext))
    return [p for p in files if p.is_file()]

def load_bg_original(path):
    try:
        return pygame.image.load(str(path)).convert_alpha()
    except Exception:
        return None

def scale_bg_to_window(bg_orig):
    if not bg_orig:
        return None
    scaled = pygame.transform.smoothscale(bg_orig, (WIDTH, HEIGHT))
    scaled.set_alpha(BG_ALPHA)
    return scaled

def random_bg_original(exclude=None):
    imgs = list_bg_images()
    if not imgs:
        return None, None
    if exclude in imgs and len(imgs) > 1:
        imgs = [p for p in imgs if p != exclude]
    choice = random.choice(imgs)
    return load_bg_original(choice), choice

# Asset helpers (cars)
_image_cache = {}

def _load_image(path, target_size):
    try:
        key = (str(path), target_size)
        if key in _image_cache:
            return _image_cache[key]
        img = pygame.image.load(str(path)).convert_alpha()
        scaled = pygame.transform.smoothscale(img, target_size)
        _image_cache[key] = scaled
        return scaled
    except Exception:
        return None

def list_enemy_sprite_files():
    if not ASSETS_DIR.exists():
        return []
    return sorted(ASSETS_DIR.glob("enemy_*.png"))

def list_special_sprite_files():
    if not ASSETS_DIR.exists():
        return []
    return sorted(ASSETS_DIR.glob("special_*.png"))

def get_player_sprite(target_size):
    path = ASSETS_DIR / "player.png"
    return _load_image(path, target_size)

# ------------------------------------------------------------------
# Entities
# ------------------------------------------------------------------
class Player:
    def __init__(self):
        self.lane = 1
        self.rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        self.rect.midbottom = (LANE_CENTERS[self.lane], HEIGHT - 24)
        self.shield = False
        self.shield_until = 0.0
        self.shoot_until = 0.0
        self.fast_until = 0.0
        self.slow_until = 0.0
        self.shoot_ready_at = 0.0
        # sprite
        self.sprite = get_player_sprite((PLAYER_W, PLAYER_H))

    def center_to_lane(self):
        self.rect.centerx = LANE_CENTERS[self.lane]

    def move_lane(self, delta):
        self.lane = clamp(self.lane + delta, 0, LANES - 1)
        self.center_to_lane()

    def move_vertical(self, dy):
        rr = road_rect()
        new_y = self.rect.y + int(dy)
        min_y = rr.top + 6
        max_y = rr.bottom - self.rect.height - 6
        self.rect.y = clamp(new_y, min_y, max_y)

    def has_power(self, kind):
        now = time.time()
        if kind == "SHIELD":
            return self.shield and self.shield_until > now
        if kind == "SHOOT":
            return self.shoot_until > now
        if kind == "FAST":
            return self.fast_until > now
        if kind == "SLOW":
            return self.slow_until > now
        return False

    def grant_power(self, kind):
        now = time.time()
        dur = POWERUP_DURATION.get(kind, 6.0)
        if kind == "SHIELD":
            self.shield = True
            self.shield_until = now + dur
        elif kind == "SHOOT":
            self.shoot_until = now + dur
            self.shoot_ready_at = now
        elif kind == "FAST":
            self.fast_until = now + dur
        elif kind == "SLOW":
            self.slow_until = now + dur

    def consume_shield_if_any(self):
        if self.has_power("SHIELD"):
            self.shield = False
            self.shield_until = 0.0
            return True
        return False

    def draw(self, surf):
        if self.sprite:
            surf.blit(self.sprite, self.rect)
        else:
            pygame.draw.rect(surf, (255, 235, 59), self.rect, border_radius=10)
        if self.has_power("SHIELD"):
            pygame.draw.circle(surf, (120, 200, 255), self.rect.center, self.rect.width, 2)

class Enemy:
    def __init__(self, lane, speed, special=False):
        self.lane = lane
        self.speed = speed
        self.special = special
        self.rect = pygame.Rect(0, 0, ENEMY_W, ENEMY_H)
        self.rect.midtop = (LANE_CENTERS[lane], -ENEMY_H - 10)
        self.counted = False
        # choose sprite
        if special:
            self.special_frames = [_load_image(p, (ENEMY_W, ENEMY_H)) for p in list_special_sprite_files()]
            if not any(self.special_frames):
                files = list_enemy_sprite_files()
                self.special_frames = [_load_image(p, (ENEMY_W, ENEMY_H)) for p in files]
            self.sprite = None
        else:
            files = list_enemy_sprite_files()
            if files:
                self.sprite = _load_image(random.choice(files), (ENEMY_W, ENEMY_H))
            else:
                self.sprite = None

    def update(self, dt, speed_scale=1.0):
        self.rect.y += int(self.speed * speed_scale * dt)

    def draw(self, surf):
        if self.special:
            frames = getattr(self, "special_frames", [])
            if frames:
                idx = int(time.time() * 6) % len(frames)
                frame = frames[idx]
                if frame:
                    surf.blit(frame, self.rect)
                    return
        if self.sprite:
            surf.blit(self.sprite, self.rect)
            return
        # fallback shape
        pygame.draw.rect(surf, (255, 85, 85) if not self.special else (255,50,50), self.rect, border_radius=10)

    def points(self):
        return SPECIAL_PASS_POINTS if self.special else NORMAL_PASS_POINTS

class PowerUp:
    TYPES = ("FAST", "SLOW", "SHIELD", "SHOOT")

    def __init__(self, lane, kind, speed):
        self.lane = lane
        self.kind = kind
        self.speed = speed
        self.rect = pygame.Rect(0, 0, PU_SIZE, PU_SIZE)
        self.rect.midtop = (LANE_CENTERS[lane], -PU_SIZE - 10)

    def update(self, dt):
        self.rect.y += int(self.speed * dt)

    def draw(self, surf):
        colors = {
            "FAST": (102, 187, 106),
            "SLOW": (66, 165, 245),
            "SHIELD": (171, 71, 188),
            "SHOOT": (255, 202, 40),
        }
        color = colors.get(self.kind, (200, 200, 200))
        pygame.draw.rect(surf, color, self.rect, border_radius=12)
        draw_text(surf, self.kind[0].upper(), (self.rect.centerx, self.rect.centery - 12), font=FONT_BIG, color=(15, 15, 15), center=True)

# ------------------------------------------------------------------
# Game
# ------------------------------------------------------------------
class Game:
    def __init__(self):
        self.bg_original = None
        self.bg_surface = None
        self.bg_path = None
        self.bg_last_level_path = None
        self.reset(full=True)

    def _set_background(self, path=None, exclude=None):
        if path is None:
            self.bg_original, self.bg_path = random_bg_original(exclude=exclude)
        else:
            self.bg_original = load_bg_original(path)
            self.bg_path = path
        self.bg_surface = scale_bg_to_window(self.bg_original)

    def _rescale_bg(self):
        self.bg_surface = scale_bg_to_window(self.bg_original)

    def reset(self, full=False):
        self.state = "START"  # START, RUN, PAUSE, GAMEOVER
        self.player = Player()
        self.enemies = []
        self.powerups = []
        self.score = 0
        self.passed = 0
        self.level = 1
        self.required = START_REQUIRED
        self.enemy_speed = BASE_ENEMY_SPEED
        self.spawn_ms = SPAWN_MS_START
        self.running_time = 0.0
        self.level_up_flash_until = 0.0
        self._set_background(path=None, exclude=None)
        self.bg_last_level_path = self.bg_path
        self.name_input = ""
        self.scores_snapshot = load_scores()
        self.shoot_beam_until = 0.0
        self.ui_buttons = {}

        # Spawning handled by time accumulator in update()
        self._spawn_accum = 0.0

    def level_up(self):
        self.level += 1
        self.required += REQUIRED_INC + max(0, self.level // 3)
        self.enemy_speed += LEVEL_SPEED_INC
        self.spawn_ms = max(SPAWN_MS_MIN, self.spawn_ms - SPAWN_DEC_PER_LEVEL)
        # Spawning handled by time accumulator in update()
        self._spawn_accum = 0.0
        self.level_up_flash_until = time.time() + 1.5
        self._set_background(path=None, exclude=self.bg_last_level_path)
        self.bg_last_level_path = self.bg_path

    def can_spawn_here(self, lane):
        for e in self.enemies:
            if e.lane == lane and e.rect.top < ENEMY_H * 1.5:
                return False
        for p in self.powerups:
            if p.lane == lane and p.rect.top < PU_SIZE * 1.5:
                return False
        return True

    def spawn(self):
        lane = random.randrange(LANES)
        if not self.can_spawn_here(lane):
            return
        if random.random() < POWERUP_SPAWN_CHANCE:
            kind = random.choice(PowerUp.TYPES)
            pu = PowerUp(lane, kind, self.enemy_speed * (0.85 if kind == "SLOW" else 1.0))
            self.powerups.append(pu)
        else:
            special = random.random() < SPECIAL_ENEMY_CHANCE
            speed = self.enemy_speed * (SPECIAL_SPEED_FACTOR if special else 1.0)
            enemy = Enemy(lane, speed, special=special)
            self.enemies.append(enemy)

    def _count_pass_and_score(self, enemy):
        self.passed += 1
        self.score += enemy.points()
        if self.passed >= self.required:
            self.level_up()

    def update(self, dt):
        if self.state != "RUN":
            return

        # continuous up/down control
        keys = pygame.key.get_pressed()
        dy = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= PLAYER_VERT_SPEED * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += PLAYER_VERT_SPEED * dt
        if dy:
            self.player.move_vertical(dy)

        # spawn by accumulator (reliable across platforms)
        self._spawn_accum += dt
        interval = max(0.05, self.spawn_ms / 1000.0)
        while self._spawn_accum >= interval:
            self.spawn()
            self._spawn_accum -= interval

        now = time.time()
        speed_scale = 1.0
        if self.player.has_power("FAST"):
            speed_scale *= 1.35
        if self.player.has_power("SLOW"):
            speed_scale *= 0.7

        # enemies
        for e in list(self.enemies):
            e.update(dt, speed_scale=speed_scale)
            if not e.counted and e.rect.top > HEIGHT:
                e.counted = True
                self.enemies.remove(e)
                self._count_pass_and_score(e)

        # powerups
        for p in list(self.powerups):
            p.update(dt * speed_scale)
            if p.rect.top > HEIGHT:
                self.powerups.remove(p)

        # collect powerups
        for p in list(self.powerups):
            if self.player.rect.colliderect(p.rect):
                self.player.grant_power(p.kind)
                self.powerups.remove(p)

        # collisions (ignore when >50% off bottom)
        safe_top = HEIGHT - ENEMY_H // 2
        for e in list(self.enemies):
            if e.rect.top > safe_top:
                continue
            if self.player.rect.colliderect(e.rect):
                if self.player.consume_shield_if_any():
                    self.enemies.remove(e)
                else:
                    self.state = "GAMEOVER"
                    break

        # shooting clears lane ahead
        if now < self.shoot_beam_until:
            lane = self.player.lane
            for e in list(self.enemies):
                if e.lane == lane and e.rect.bottom < self.player.rect.top:
                    self.enemies.remove(e)
                    self._count_pass_and_score(e)

        self.running_time += dt

    def try_fire(self):
        now = time.time()
        if not self.player.has_power("SHOOT"):
            return
        if now < self.player.shoot_ready_at:
            return
        self.shoot_beam_until = now + 0.22
        self.player.shoot_ready_at = now + SHOOT_COOLDOWN

    # ---------- Panels ----------
    def _draw_left_panel(self, surf):
        panel = pygame.Rect(0, 0, PANEL_LEFT_W, HEIGHT)
        s = pygame.Surface(panel.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 110))
        surf.blit(s, panel.topleft)
        draw_text(surf, "FRAMERACING", (panel.centerx, 16), font=PANEL_TITLE_FONT, center=True)
        draw_text(surf, f"Score: {self.score}", (12, 56), font=PANEL_FONT)
        draw_text(surf, f"Level: {self.level}", (12, 78), font=PANEL_FONT)
        draw_text(surf, f"Passed: {self.passed}/{self.required}", (12, 100), font=PANEL_FONT)
        draw_text(surf, "Power-ups:", (12, 130), font=PANEL_TITLE_FONT)
        y = 152
        powers = []
        if self.player.has_power("FAST"): powers.append("FAST")
        if self.player.has_power("SLOW"): powers.append("SLOW")
        if self.player.has_power("SHIELD"): powers.append("BUBBLE")
        if self.player.has_power("SHOOT"): powers.append("SHOOT")
        if powers:
            for p in powers:
                draw_text(surf, f"• {p}", (18, y), font=PANEL_FONT); y += 20
        else:
            draw_text(surf, "• none", (18, y), font=PANEL_FONT)
        draw_text(surf, "Controls", (panel.centerx, HEIGHT - 210), font=PANEL_TITLE_FONT, center=True)
        for i, t in enumerate(["←/→ lanes", "↑/↓ up/down", "SPACE shoot", "P pause", "ESC exit", "+20 normal", "+50 special"]):
            draw_text(surf, t, (12, HEIGHT - 186 + i*20), font=PANEL_FONT)

    def _draw_right_panel(self, surf):
        panel = pygame.Rect(WIDTH - PANEL_RIGHT_W, 0, PANEL_RIGHT_W, HEIGHT)
        s = pygame.Surface(panel.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 110))
        surf.blit(s, panel.topleft)
        # Buttons
        self.ui_buttons = {}
        btn_y = 10
        btn_w, btn_h = 68, 28
        gap = 12
        labels = [("Start", "start"), ("Pause", "pause"), ("Exit", "exit")]
        x = panel.left + 12
        for label, key in labels:
            rect = pygame.Rect(x, btn_y, btn_w, btn_h)
            pygame.draw.rect(surf, (230, 230, 230), rect, border_radius=8)
            draw_text(surf, label, rect.center, center=True, color=(20, 20, 20))
            self.ui_buttons[key] = rect
            x += btn_w + gap
        # High scores
        draw_text(surf, "Top 10", (panel.centerx, 58), font=PANEL_TITLE_FONT, center=True)
        scores = load_scores()
        y = 86
        for i, srow in enumerate(scores[:10]):
            txt = f"{i+1:>2}. {srow.get('name','PLAYER')[:12]:<12} {srow.get('score',0):>5}"
            draw_text(surf, txt, (panel.left + 12, y), font=PANEL_FONT)
            y += 20

    def draw(self, surf):
        surf.fill(ROAD_COLOR)

        # road area first (underlay for background overlay)
        rr = road_rect()
        pygame.draw.rect(surf, (30, 30, 34), rr)

        # background overlay across the whole window
        if self.bg_surface:
            surf.blit(self.bg_surface, (0, 0))

        # panels
        self._draw_left_panel(surf)
        self._draw_right_panel(surf)

        # lane markers
        for i in range(LANES + 1):
            x = rr.left + ROAD_MARGIN + int((rr.width - 2 * ROAD_MARGIN) * i / LANES)
            pygame.draw.line(surf, LANE_COLOR, (x, 0), (x, HEIGHT), 2)

        # entities
        for e in self.enemies:
            e.draw(surf)
        for p in self.powerups:
            p.draw(surf)
        self.player.draw(surf)

        
        # shoot beam (laser line)
        if time.time() < self.shoot_beam_until:
            lane = self.player.lane
            y0 = 0
            y1 = self.player.rect.top - 6
            cx = LANE_CENTERS[lane]
            # draw a soft glow + bright core using alpha surfaces
            glow_w = 16
            core_w = 4
            glow_h = max(1, y1 - y0)
            # glow
            gsurf = pygame.Surface((glow_w, glow_h), pygame.SRCALPHA)
            gsurf.fill((255, 255, 255, 60))
            surf.blit(gsurf, (cx - glow_w // 2, y0))
            # core
            csurf = pygame.Surface((core_w, glow_h), pygame.SRCALPHA)
            csurf.fill((255, 255, 255, 230))
            surf.blit(csurf, (cx - core_w // 2, y0))

        

        if time.time() < self.level_up_flash_until:
            draw_text(surf, "LEVEL UP!", (rr.centerx, HEIGHT//2 - 120), font=FONT_HUGE, color=(255, 255, 255), center=True)

        if self.state == "PAUSE":
            draw_text(surf, "PAUSED", (rr.centerx, HEIGHT//2 - 24), font=FONT_HUGE, color=(255, 255, 255), center=True)

        if self.state == "START":
            self.draw_start(surf)

        if self.state == "GAMEOVER":
            self.draw_gameover(surf)

    def draw_start(self, surf):
        rr = road_rect()
        draw_text(surf, "FRAMERACING", (rr.centerx, HEIGHT//2 - 220), font=FONT_HUGE, color=(255, 255, 255), center=True)
        lines = [
            "Avoid cars, pass as many as you can.",
            "Power-ups: FAST, SLOW, BUBBLE, SHOOT",
            "SPACE to shoot when SHOOT is active",
            "←/→ lane, ↑/↓ up/down, P pause, ESC quit",
            "+20 per normal car, +50 special",
            "Press ENTER to start",
        ]
        for i, t in enumerate(lines):
            draw_text(surf, t, (rr.centerx, HEIGHT//2 - 80 + i*32), center=True)

    def draw_gameover(self, surf):
        rr = road_rect()
        draw_text(surf, "GAME OVER", (rr.centerx, HEIGHT//2 - 140), font=FONT_HUGE, color=(255, 255, 255), center=True)
        draw_text(surf, f"Score: {self.score}", (rr.centerx, HEIGHT//2 - 80), font=FONT_BIG, center=True)
        draw_text(surf, "Enter your name:", (rr.centerx, HEIGHT//2 - 10), font=FONT_BIG, center=True)
        name_display = self.name_input if self.name_input else "PLAYER"
        pygame.draw.rect(surf, (240, 240, 240), pygame.Rect(rr.centerx - 150, HEIGHT//2 + 24, 300, 40), 2, border_radius=8)
        draw_text(surf, name_display, (rr.centerx, HEIGHT//2 + 30), center=True)
        draw_text(surf, "Press ENTER to save, R to restart", (rr.centerx, HEIGHT//2 + 90), center=True)

    # Events
    def handle_event(self, event):
        global WIDTH, HEIGHT, screen
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = max(640, event.w), max(480, event.h)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            recalc_geometry(self.player)
            self._rescale_bg()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if "pause" in self.ui_buttons and self.ui_buttons["pause"].collidepoint(pos):
                if self.state == "RUN":
                    self.state = "PAUSE"
                elif self.state == "PAUSE":
                    self.state = "RUN"
                return
            if "start" in self.ui_buttons and self.ui_buttons["start"].collidepoint(pos):
                if self.state in ("START", "GAMEOVER"):
                    self.reset(full=False)
                    self.state = "RUN"
                elif self.state == "PAUSE":
                    self.state = "RUN"
                return
            if "exit" in self.ui_buttons and self.ui_buttons["exit"].collidepoint(pos):
                pygame.quit()
                sys.exit(0)
                return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)

            if self.state == "RUN":
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.player.move_lane(-1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.player.move_lane(+1)
                elif event.key == pygame.K_SPACE:
                    self.try_fire()
                elif event.key == pygame.K_p:
                    self.state = "PAUSE"
            elif self.state == "PAUSE":
                if event.key == pygame.K_p:
                    self.state = "RUN"
            elif self.state == "START":
                if event.key == pygame.K_RETURN:
                    self.state = "RUN"
            elif self.state == "GAMEOVER":
                if event.key == pygame.K_RETURN:
                    name = self.name_input.strip() or "PLAYER"
                    save_score(name, self.score)
                    self.reset(full=False)
                elif event.key == pygame.K_r:
                    self.reset(full=False)
                elif event.key == pygame.K_BACKSPACE:
                    self.name_input = self.name_input[:-1]
                else:
                    ch = event.unicode
                    if ch and (ch.isalnum() or ch in (" ", "_", "-")) and len(self.name_input) < 16:
                        self.name_input += ch

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    game = Game()
    dt = 0.0
    WIDTH, HEIGHT = screen.get_size()
    recalc_geometry(game.player)
    game._rescale_bg()
    while True:
        for event in pygame.event.get():
            game.handle_event(event)

        if game.state == "RUN":
            game.update(dt)

        game.draw(screen)
        pygame.display.flip()
        dt = clock.tick(FPS) / 1000.0

if __name__ == "__main__":
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    main()
