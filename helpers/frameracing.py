#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrameRacing — retro 4-lane racer (Pygame)

Paths
- Game: root/helpers/frameracing.py
- High scores: root/presets/setsave/racescore.json
- Backgrounds: root/presets/startup/  (15% opacity overlay, changes on level-up)
- Car sprites: root/assets/cars/  (player.png, enemy_*.png, special_*.png)

Notes
- Normal pass: +20 points; Special pass: +50 points
- Special cars are faster and can be animated (multiple special_*.png frames)
- Enemy cars become harmless once they are >50% past the player car vertically
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
import warnings
import math
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)

import pygame

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[1]
HELPERS_DIR = ROOT_DIR / "helpers"
SCORES_PATH = ROOT_DIR / "presets" / "setsave" / "racescore.json"
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

# Moving dashed lane divider config
LANE_DASH_COLOR = (235, 235, 235)  # bright dashed lane dividers
LANE_DASH_LEN = 42  # pixels per dash segment
LANE_DASH_GAP = 28  # pixels between dashes
LANE_DASH_THICKNESS = 4  # base thickness; scaled with lane width
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


# Difficulty ramp (spawn interval easing + density clamp)
DIFF_RAMP_SEC = 120.0   # seconds to ramp from SPAWN_MS_START toward SPAWN_MS_MIN
MAX_ENEMIES_PER_LANE = 3  # clamp on-screen density (enemies)
MAX_POWERUPS_ONSCREEN = 3  # cap power-ups on screen


# Power-ups
POWERUP_SPAWN_CHANCE = 0.12
POWERUP_DURATION = {
    "FAST": 7.0,
    "SLOW": 5.0,
    "SHIELD": 12.0,
    "SHOOT": 8.0,
}
SHOOT_COOLDOWN = 0.5  # seconds



# FAST stacking config
FAST_BASE_MULT = 1.35     # first FAST stack
FAST_STACK_BONUS = 0.10   # extra multiplier per additional concurrent FAST
# Scoring & special enemies
NORMAL_PASS_POINTS = 20
SPECIAL_PASS_POINTS = 50
SPECIAL_ENEMY_CHANCE = 0.18
SPECIAL_SPEED_FACTOR = 1.35

# Extra life award
LIFE_SCORE_STEP = 5000
CONFETTI_DURATION = 1.6
FIREWORKS_DURATION = 2.8
CONFETTI_COUNT = 140
FIREWORK_BURSTS = 12
FIREWORK_PARTICLES = 48

# Background
BG_ALPHA = int(255 * 0.35)  # 55% opacity

# HUD
HUD_RING_RADIUS = 42
HUD_RING_WIDTH = 8
HUD_TEXT_COLOR = (250, 250, 250)
HUD_GHOST_COLOR = (255, 255, 255, 40)
HUD_RING_COLOR = (255, 255, 255, 220)
HUD_BOUNCE_DUR = 0.25  # seconds
HUD_BOUNCE_MAX = 0.18  # extra scale at start

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

def recalc_geometry(player=None, enemies=None, powerups=None):
    global LANE_CENTERS, LANE_RECTS, LANE_W
    LANE_CENTERS, LANE_RECTS, LANE_W = lanes_geometry()
    # Keep entities centered to their lanes after any window resize so they remain grabbable.
    if player is not None:
        # Keep X centered to lane; preserve Y within road bounds.
        player.center_to_lane()
        rr = road_rect()
        player.rect.y = clamp(player.rect.y, rr.top + 6, rr.bottom - player.rect.height - 6)
    if enemies is not None:
        for e in enemies:
            try:
                e.rect.centerx = LANE_CENTERS[e.lane]
            except Exception:
                pass
    if powerups is not None:
        for p in powerups:
            try:
                p.rect.centerx = LANE_CENTERS[p.lane]
            except Exception:
                pass


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def lerp(a, b, t):
    return a + (b - a) * t

def ease_out_cubic(t):
    t = clamp01(t)
    return 1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t)


def draw_text(surf, text, pos, font=FONT, color=(240, 240, 240), center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surf.blit(img, rect)
    return rect

def draw_heart(surf, pos, size=10, color=(220,50,80)):
    # simple vector heart: two circles + a triangle-ish bottom
    import pygame
    x, y = pos
    r = size
    # top bumps
    pygame.draw.circle(surf, color, (int(x + r*0.6), int(y - r*0.2)), int(r*0.6))
    pygame.draw.circle(surf, color, (int(x + r*1.4), int(y - r*0.2)), int(r*0.6))
    # bottom
    pts = [
        (x, y),
        (x + r*2, y),
        (x + r, y + r*1.7),
    ]
    pygame.draw.polygon(surf, color, [(int(px), int(py)) for px,py in pts])


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
def is_top10(score):
    scores = load_scores()
    if len(scores) < 10:
        return True
    # Allow ties to qualify
    try:
        min_top = int(scores[-1].get("score", 0))
    except Exception:
        min_top = 0
    return int(score) >= min_top

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

        # --- Stacking power-ups ---
        # Each kind maps to a list of expiry timestamps for active stacks.
        # Example: {'FAST': [t1, t2], 'SLOW':[t3], ...}
        self.powers = {k: [] for k in ("FAST", "SLOW", "SHIELD", "SHOOT")}
        self.shoot_ready_at = 0.0

        # sprite
        self.sprite = get_player_sprite((PLAYER_W, PLAYER_H))

        # particle store (used by effects helpers)
        self.particles = []
        self._pt_last_time = time.time()

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

    # --- Power helpers (stack management) ---
    def _purge_expired(self):
        now = time.time()
        for k in list(self.powers.keys()):
            stacks = [t for t in self.powers[k] if t > now]
            self.powers[k] = stacks

    def active_stacks(self, kind):
        self._purge_expired()
        return len(self.powers.get(kind, []))

    def has_power(self, kind):
        return self.active_stacks(kind) > 0

    def grant_power(self, kind):
        now = time.time()
        dur = POWERUP_DURATION.get(kind, 6.0)
        expiry = now + dur
        self.powers.setdefault(kind, []).append(expiry)
        if kind == "SHOOT":
            # allow immediate use if not on cooldown
            self.shoot_ready_at = now if not self.shoot_ready_at else min(self.shoot_ready_at, now)

    def consume_shield_if_any(self):
        # consume one shield stack (if any) on collision
        if self.active_stacks("SHIELD") > 0:
            # remove the soonest-expiring one
            self.powers["SHIELD"].sort()
            self.powers["SHIELD"].pop(0)
            return True
        return False

    def time_remaining_max(self, kind):
        """Return the maximum remaining seconds among stacks for UI bars."""
        self._purge_expired()
        now = time.time()
        stacks = self.powers.get(kind, [])
        if not stacks:
            return 0.0
        return max(0.0, max(stacks) - now)

    # ---- visual helpers (kept from your newer build) ----
    def _spawn_confetti(self, cx, cy, count=CONFETTI_COUNT):
        import pygame, random, math, time
        for _ in range(count):
            ang = random.uniform(0, 2*math.pi)
            spd = random.uniform(80, 220)
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            size = random.randint(2, 4)
            color = random.choice([(240,60,60),(60,180,255),(255,220,60),(120,220,120),(200,120,255)])
            self.particles.append({
                'type':'confetti','x':cx,'y':cy,'vx':vx,'vy':vy,'g':180.0,'life':CONFETTI_DURATION,'size':size,'color':color
            })

    def _spawn_fireworks(self, bursts=FIREWORK_BURSTS, per=FIREWORK_PARTICLES):
        import pygame, random, math
        for _ in range(bursts):
            cx = random.uniform(80, WIDTH-80)
            cy = random.uniform(60, HEIGHT-140)
            base_col = random.choice([(255,120,120),(120,255,200),(255,240,120),(140,180,255),(255,150,220)])
            for i in range(per):
                ang = (i/per) * 2*math.pi + random.uniform(-0.05,0.05)
                spd = random.uniform(60, 260)
                vx = math.cos(ang) * spd
                vy = math.sin(ang) * spd
                size = random.randint(2, 3)
                color = base_col
                self.particles.append({
                    'type':'firework','x':cx,'y':cy,'vx':vx,'vy':vy,'g':0.0,'life':FIREWORKS_DURATION,'size':size,'color':color
                })

    def _update_and_draw_particles(self, surf):
        import pygame, time
        now = time.time()
        dt = max(0.0, min(0.05, now - getattr(self, '_pt_last_time', now)))
        self._pt_last_time = now
        if not self.particles:
            return
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        alive = []
        for p in self.particles:
            life = p['life'] - dt
            if life <= 0:
                continue
            p['life'] = life
            # physics
            p['vy'] += p['g'] * dt
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            # alpha based on remaining life
            alpha = max(0, min(255, int(255 * (life / (CONFETTI_DURATION if p['type']== 'confetti' else FIREWORKS_DURATION)) )))
            col = (*p['color'], alpha)
            s = p['size']
            if p['type'] == 'confetti':
                pygame.draw.rect(overlay, col, (int(p['x']), int(p['y']), s, s))
            else:
                pygame.draw.circle(overlay, col, (int(p['x']), int(p['y'])), s)
            alive.append(p)
        self.particles = alive
        surf.blit(overlay, (0,0))

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


    def _density_at_cap(self):

        # Check global density caps for enemies and powerups

        if len(self.enemies) >= MAX_ENEMIES_PER_LANE * LANES:

            return True

        if len(self.powerups) >= MAX_POWERUPS_ONSCREEN:

            return True

        return False



    def current_spawn_interval(self):

        """Return current spawn interval (seconds) using a time-eased curve

        combined with level-based adjustments. This keeps dt-based spawning

        stable while ramping difficulty smoothly over time."""

        # Level baseline (ms) clamped to configured min/max

        base_ms = clamp(SPAWN_MS_MIN, SPAWN_MS_START - (self.level - 1) * SPAWN_DEC_PER_LEVEL, SPAWN_MS_START)

        # Time-based easing from base_ms towards SPAWN_MS_MIN over DIFF_RAMP_SEC

        r = clamp01(self.running_time / max(0.001, DIFF_RAMP_SEC))

        eased = ease_out_cubic(r)

        ms = lerp(base_ms, SPAWN_MS_MIN, eased)

        # Hard lower bound and minimum safety interval

        return max(0.05, ms / 1000.0)


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
        self.particles = []
        self.confetti_until = 0.0
        self.fireworks_until = 0.0
        self._pt_last_time = time.time()
        self.lives = 0
        self.next_life_at = LIFE_SCORE_STEP
        self.extra_life_flash_until = 0.0
        self._set_background(path=None, exclude=None)
        self.bg_last_level_path = self.bg_path
        self.name_input = ""
        self.name_allowed = False
        self.scores_snapshot = load_scores()
        self.shoot_beam_until = 0.0
        self.ui_buttons = {}
        # lane divider scroll
        self.lane_dash_offset = 0.0
        self.shake_until = 0.0
        self.shake_mag = 0
        self.trail = []
        self.trail_block_until = 0.0
        self.top10_flash_until = 0.0

        # Combo system (for UI + scoring multipliers)
        self.combo_window = 3.0   # seconds to chain a pass
        self.combo_max = 5        # cap multiplier at x5
        self.combo_count = 0
        self.combo_mult = 1
        self.combo_timer = 0.0

        # HUD bounce state
        self.hud_bounce_t = 999.0
        self.hud_bounce_dur = HUD_BOUNCE_DUR
        self.last_speed_mult = 1.0
        self.last_combo_mult = 1

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
        if self._density_at_cap():
            return
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
        # Called when an enemy was safely passed or cleared.
        # Update combo before awarding points.
        if self.combo_timer > 0:
            self.combo_count += 1
        else:
            self.combo_count = 1
        self.combo_mult = 1 + min(self.combo_count - 1, self.combo_max - 1)
        self.combo_timer = self.combo_window

        # Award points with multiplier
        self.passed += 1
        self.score += enemy.points() * self.combo_mult
        # Award extra lives every LIFE_SCORE_STEP points
        while self.score >= self.next_life_at:
            threshold = self.next_life_at
            self.lives += 1
            self.next_life_at += LIFE_SCORE_STEP
            self.extra_life_flash_until = time.time() + 1.2
            self.hud_bounce_t = 0.0
            # visuals: confetti at first 5000, fireworks for 10k and beyond
            if threshold <= LIFE_SCORE_STEP:
                self.confetti_until = time.time() + CONFETTI_DURATION
                self._spawn_confetti(self.player.rect.centerx, self.player.rect.centery)
            else:
                self.fireworks_until = time.time() + FIREWORKS_DURATION
                self._spawn_fireworks()

        # Bounce HUD on combo change
        if self.combo_mult != self.last_combo_mult:
            self.hud_bounce_t = 0.0

        if self.passed >= self.required:
            self.level_up()

    def _current_speed_mult(self):
        m = 1.0
        fast_n = self.player.active_stacks("FAST")
        slow_n = self.player.active_stacks("SLOW")
        if fast_n > 0:
            m *= (FAST_BASE_MULT + max(0, fast_n - 1) * FAST_STACK_BONUS)
        if slow_n > 0:
            m *= 0.7  # single slow effect regardless of stack count
        return m

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

        

        # purge expired power-up stacks
        self.player._purge_expired()

        # spawn by accumulator (reliable across platforms) with eased interval + density clamp

        self._spawn_accum += dt

        interval = self.current_spawn_interval()

        # Use a while-loop to catch up if we were paused or frames dropped, but respect density cap

        safety = 0

        while self._spawn_accum >= interval and safety < 8:

            safety += 1

            if self._density_at_cap():

                # hold at most one interval worth of debt to avoid runaway accumulation at cap

                self._spawn_accum = min(self._spawn_accum, interval)

                break

            self.spawn()

            self._spawn_accum -= interval


        now = time.time()
        speed_scale = 1.0
        if self.player.has_power("FAST"):
            speed_scale *= 1.35
        if self.player.has_power("SLOW"):
            speed_scale *= 0.7

        # scroll dashed lane dividers at game speed
        self.lane_dash_offset = (self.lane_dash_offset + (self.enemy_speed * speed_scale) * dt) % (LANE_DASH_LEN + LANE_DASH_GAP)
        # FAST ghost trail sampling (disabled while holding BACK keys)
        nowt = time.time()
        keys = pygame.key.get_pressed()
        back_held = keys[pygame.K_DOWN] or keys[pygame.K_s]
        if self.player.has_power('FAST') and not back_held:
            self.trail.append({'x': self.player.rect.centerx, 'y': self.player.rect.centery, 't': nowt})
        else:
            self.trail = []
        self.trail_block_until = 0.0
        self.trail = [t for t in self.trail if nowt - t['t'] <= 0.4]

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

        # collisions (enemy stops hurting once it's >50% past the player's car)
        safe_top = self.player.rect.centery
        for e in list(self.enemies):
            if e.rect.top > safe_top:
                continue
            if self.player.rect.colliderect(e.rect):
                if self.player.consume_shield_if_any():
                    self.enemies.remove(e)
                    self._spawn_sparks(self.player.rect.centerx, self.player.rect.centery)
                    self._trigger_shake(5, 0.18)
                else:
                    if self.lives > 0:
                        self.lives -= 1
                        # remove the colliding enemy and keep going
                        self.enemies.remove(e)
                        self.hud_bounce_t = 0.0
                        self._spawn_sparks(self.player.rect.centerx, self.player.rect.centery)
                        self._trigger_shake(7, 0.22)
                    else:
                        self.state = "GAMEOVER"
                        self.name_allowed = is_top10(self.score)
                        self._spawn_sparks(self.player.rect.centerx, self.player.rect.centery)
                        self._trigger_shake(10, 0.35)
                        if self.name_allowed:
                            self._spawn_confetti(self.player.rect.centerx, self.player.rect.centery)
                            self.top10_flash_until = time.time() + 1.5
                        break

        # shooting clears lane ahead
        if now < self.shoot_beam_until:
            lane = self.player.lane
            for e in list(self.enemies):
                if e.lane == lane and e.rect.bottom < self.player.rect.top:
                    self.enemies.remove(e)
                    self._count_pass_and_score(e)

        # Combo timer countdown + reset when expired
        if self.combo_timer > 0.0:
            self.combo_timer -= dt
            if self.combo_timer <= 0.0:
                self.combo_timer = 0.0
                self.combo_count = 0
                if self.combo_mult != 1:
                    # trigger a gentle bounce on reset
                    self.hud_bounce_t = 0.0
                self.combo_mult = 1

        # HUD bounce trigger on speed change (FAST/SLOW toggles)
        cur_speed_mult = round(self._current_speed_mult(), 2)
        if cur_speed_mult != self.last_speed_mult:
            self.hud_bounce_t = 0.0

        # advance bounce time
        self.hud_bounce_t = min(self.hud_bounce_t + dt, self.hud_bounce_dur + 0.001)

        # update last-knowns for change detection next frame
        self.last_speed_mult = cur_speed_mult
        self.last_combo_mult = self.combo_mult

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
        # Show extra lives alongside power-ups
        if self.lives > 0:
            draw_heart(surf, (18, y+10), 10)
            draw_text(surf, f"× {self.lives}", (36, y), font=PANEL_FONT)
            y += 20
        # Show active powers with stack counts
        for kind, label in (("FAST","FAST"), ("SLOW","SLOW"), ("SHIELD","BUBBLE"), ("SHOOT","SHOOT")):
            n = self.player.active_stacks(kind)
            if n > 0:
                powers.append(f"• {label} ×{n}")
        if powers:
            for p in powers:
                draw_text(surf, p, (18, y), font=PANEL_FONT); y += 20
        else:
            draw_text(surf, "• none", (18, y), font=PANEL_FONT)
        draw_text(surf, "Controls", (panel.centerx, HEIGHT - 210), font=PANEL_TITLE_FONT, center=True)
        for i, t in enumerate(["←/→ lanes", "↑/↓ up/down", "SPACE shoot", "P/ESC pause", "Exit button quits", "+20 normal", "+50 special"]):
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

    
    def _spawn_confetti(self, cx, cy, count=CONFETTI_COUNT):
        import pygame, random, math, time
        for _ in range(count):
            ang = random.uniform(0, 2*math.pi)
            spd = random.uniform(80, 220)
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            size = random.randint(2, 4)
            color = random.choice([(240,60,60),(60,180,255),(255,220,60),(120,220,120),(200,120,255)])
            self.particles.append({
                'type':'confetti','x':cx,'y':cy,'vx':vx,'vy':vy,'g':180.0,'life':CONFETTI_DURATION,'size':size,'color':color
            })

    def _spawn_fireworks(self, bursts=FIREWORK_BURSTS, per=FIREWORK_PARTICLES):
        import pygame, random, math
        for _ in range(bursts):
            cx = random.uniform(80, WIDTH-80)
            cy = random.uniform(60, HEIGHT-140)
            base_col = random.choice([(255,120,120),(120,255,200),(255,240,120),(140,180,255),(255,150,220)])
            for i in range(per):
                ang = (i/per) * 2*math.pi + random.uniform(-0.05,0.05)
                spd = random.uniform(60, 260)
                vx = math.cos(ang) * spd
                vy = math.sin(ang) * spd
                size = random.randint(2, 3)
                color = base_col
                self.particles.append({
                    'type':'firework','x':cx,'y':cy,'vx':vx,'vy':vy,'g':0.0,'life':FIREWORKS_DURATION,'size':size,'color':color
                })

    def _update_and_draw_particles(self, surf):
        import pygame, time
        now = time.time()
        dt = max(0.0, min(0.05, now - getattr(self, '_pt_last_time', now)))
        self._pt_last_time = now
        if not getattr(self, 'particles', None):
            return
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        alive = []
        for p in self.particles:
            life = p['life'] - dt
            if life <= 0:
                continue
            p['life'] = life
            p['vy'] += p['g'] * dt
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            alpha = max(0, min(255, int(255 * (life / (CONFETTI_DURATION if p['type']== 'confetti' else FIREWORKS_DURATION)) )))
            col = (*p['color'], alpha)
            s = p['size']
            if p['type'] == 'confetti':
                pygame.draw.rect(overlay, col, (int(p['x']), int(p['y']), s, s))
            elif p['type'] == 'spark':
                pygame.draw.circle(overlay, col, (int(p['x']), int(p['y'])), max(1, s))
            elif p['type'] == 'puff':
                pygame.draw.circle(overlay, (255,255,255,int(alpha*0.4)), (int(p['x']), int(p['y'])), max(3, s))
            else:
                pygame.draw.circle(overlay, col, (int(p['x']), int(p['y'])), s)
            alive.append(p)
        self.particles = alive
        surf.blit(overlay, (0,0))
    def _trigger_shake(self, mag=6, dur=0.25):
        import time
        self.shake_mag = int(max(1, mag))
        self.shake_until = time.time() + max(0.05, dur)

    def _spawn_sparks(self, cx, cy, count=28):
        import random, math
        for _ in range(count):
            ang = random.uniform(0, 2*math.pi)
            spd = random.uniform(220, 420)
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            size = 2
            color = random.choice([(255,220,120),(255,160,80),(255,255,255)])
            self.particles.append({'type':'spark','x':cx,'y':cy,'vx':vx,'vy':vy,'g':0.0,'life':0.35,'size':size,'color':color})
    def _draw_ghost_trail(self, surf):
        import pygame, time
        if not self.trail:
            return
        now = time.time()
        self.trail = [t for t in self.trail if now - t['t'] <= 0.4]
        for t in self.trail:
            alpha = int(255 * max(0.0, 1.0 - (now - t['t']) / 0.4))
            if self.player.sprite:
                ghost = self.player.sprite.copy()
                ghost.set_alpha(alpha)
                rect = ghost.get_rect(center=(t['x'], t['y']))
                surf.blit(ghost, rect.topleft)
            else:
                r = self.player.rect.copy()
                r.center = (t['x'], t['y'])
                gsurf = pygame.Surface(r.size, pygame.SRCALPHA)
                gsurf.fill((255,255,255,alpha))
                surf.blit(gsurf, r.topleft)

    def _draw_power_bar(self, surf):
        import pygame, time
        rr = road_rect()
        pad = 8
        y = 8
        x = rr.left + pad
        h = 34
        bg = pygame.Surface((rr.width - pad*2, h), pygame.SRCALPHA)
        bg.fill((0,0,0,110))
        surf.blit(bg, (x, y))
        items = []
        # Prepare list with label (including stack count), remaining secs, and color
        kind_info = [
            ("FAST", "F", (102, 187, 106)),
            ("SLOW", "S", (66, 165, 245)),
            ("SHIELD", "B", (171, 71, 188)),
            ("SHOOT", "!", (255, 202, 40)),
        ]
        for kind, sym, col in kind_info:
            n = self.player.active_stacks(kind)
            if n > 0:
                secs = self.player.time_remaining_max(kind)
                label = f"{sym}×{n}" if n > 1 else sym
                items.append((label, secs, col, kind))
        if items:
            ix = x + 10
            for label, secs, col, kind in items:
                w = 92
                rect = pygame.Rect(ix, y+4, w, h-8)
                pygame.draw.rect(surf, col, rect, border_radius=8)
                dur = POWERUP_DURATION.get(kind, 8.0)
                pct = max(0.0, min(1.0, secs / max(0.001, dur)))
                tb = pygame.Rect(rect.left+4, rect.bottom-8, int((rect.width-8)*pct), 4)
                pygame.draw.rect(surf, (25,25,25), tb, border_radius=2)
                draw_text(surf, f"{label}", rect.center, font=FONT_BIG, color=(20,20,20), center=True)
                draw_text(surf, f"{secs:0.1f}s", (rect.right+4, rect.centery-10), font=PANEL_FONT)
                ix += w + 52
        speed_mult = self._current_speed_mult()
        speed_val = int(self.enemy_speed * speed_mult)
        draw_text(surf, f"SPD {speed_val}", (rr.right - 120, y+8), font=PANEL_FONT)
        draw_text(surf, f"Combo x{self.combo_mult}", (rr.right - 140, y+20), font=PANEL_FONT)

    
    def draw(self, surf):
        import pygame, random, time
        world = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        world.fill(ROAD_COLOR)
        rr = road_rect()
        pygame.draw.rect(world, (30, 30, 34), rr)
        if self.bg_surface:
            world.blit(self.bg_surface, (0, 0))
        # lane markers
        xs = [rr.left + ROAD_MARGIN + int((rr.width - 2 * ROAD_MARGIN) * i / LANES) for i in range(LANES + 1)]
        pygame.draw.line(world, LANE_COLOR, (xs[0], 0), (xs[0], HEIGHT), 2)
        pygame.draw.line(world, LANE_COLOR, (xs[-1], 0), (xs[-1], HEIGHT), 2)
        thick = clamp(int(LANE_W * 0.04), 2, LANE_DASH_THICKNESS)
        start_y = -LANE_DASH_LEN + int(self.lane_dash_offset)
        step = LANE_DASH_LEN + LANE_DASH_GAP
        for x in xs[1:-1]:
            y = start_y
            while y < HEIGHT:
                rect = pygame.Rect(x - thick // 2, y, thick, LANE_DASH_LEN)
                if rect.bottom > 0 and rect.top < HEIGHT:
                    pygame.draw.rect(world, LANE_DASH_COLOR, rect, border_radius=thick//2)
                y += step
        for e in self.enemies:
            e.draw(world)
        for p in self.powerups:
            p.draw(world)
        self._draw_ghost_trail(world)
        self.player.draw(world)
        if time.time() < self.shoot_beam_until:
            lane = self.player.lane
            y0 = 0
            y1 = self.player.rect.top - 6
            cx = LANE_CENTERS[lane]
            glow_w = 16
            core_w = 4
            glow_h = max(1, y1 - y0)
            gsurf = pygame.Surface((glow_w, glow_h), pygame.SRCALPHA)
            gsurf.fill((255, 255, 255, 60))
            world.blit(gsurf, (cx - glow_w // 2, y0))
            csurf = pygame.Surface((core_w, glow_h), pygame.SRCALPHA)
            csurf.fill((255, 255, 255, 230))
            world.blit(csurf, (cx - core_w // 2, y0))
        if time.time() < self.level_up_flash_until:
            draw_text(world, "LEVEL UP!", (rr.centerx, HEIGHT//2 - 120), font=FONT_HUGE, color=(255, 255, 255), center=True)
        if time.time() < self.extra_life_flash_until:
            draw_text(world, "EXTRA LIFE!", (rr.centerx, HEIGHT//2 - 180), font=FONT_HUGE, color=(255, 255, 255), center=True)
        self._update_and_draw_particles(world)
        ox = oy = 0
        if time.time() < getattr(self, 'shake_until', 0):
            m = max(1, getattr(self, 'shake_mag', 4))
            import random as _r
            ox = _r.randint(-m, m)
            oy = _r.randint(-m, m)
        surf.fill(ROAD_COLOR)
        surf.blit(world, (ox, oy))
        self._draw_left_panel(surf)
        self._draw_right_panel(surf)
        self._draw_power_bar(surf)
        self.draw_hud(surf)
        if self.state == "PAUSE":
            draw_text(surf, "PAUSED", (rr.centerx, HEIGHT//2 - 24), font=FONT_HUGE, color=(255, 255, 255), center=True)
        if self.state == "START":
            self.draw_start(surf)
        if self.state == "GAMEOVER":
            self.draw_gameover(surf)

    def draw_hud(self, surf):
        rr = road_rect()
        center = (rr.centerx, 80)

        # ring surface (with alpha for soft look)
        r = HUD_RING_RADIUS
        w = HUD_RING_WIDTH
        ring_surf = pygame.Surface((r*2+6, r*2+6), pygame.SRCALPHA)

        # ghost background circle
        pygame.draw.circle(ring_surf, HUD_GHOST_COLOR, (r+3, r+3), r, w)

        # combo timer arc
        if self.combo_timer > 0.0 and self.combo_window > 0.0:
            frac = max(0.0, min(1.0, self.combo_timer / self.combo_window))
            rect = pygame.Rect(3, 3, r*2, r*2)
            start = -math.pi / 2  # top
            end = start + (2 * math.pi * frac)
            pygame.draw.arc(ring_surf, HUD_RING_COLOR, rect, start, end, w)

        # bounce scale
        if self.hud_bounce_t < self.hud_bounce_dur:
            t = self.hud_bounce_t / self.hud_bounce_dur
            scale = 1.0 + HUD_BOUNCE_MAX * (1 - t) * (1 - t)  # ease-out quad
        else:
            scale = 1.0

        # blit ring
        rs = ring_surf
        if abs(scale - 1.0) > 1e-3:
            size = (int(rs.get_width()*scale), int(rs.get_height()*scale))
            rs = pygame.transform.smoothscale(rs, size)
        rs_rect = rs.get_rect(center=center)
        surf.blit(rs, rs_rect)

        # text: speed multiplier / combo multiplier (minimal)
        speed_mult = self._current_speed_mult()
        text = f"{speed_mult:.2f}× / x{self.combo_mult}"
        img = FONT_BIG.render(text, True, HUD_TEXT_COLOR)
        if abs(scale - 1.0) > 1e-3:
            tw, th = img.get_size()
            img = pygame.transform.smoothscale(img, (int(tw*scale), int(th*scale)))
        surf.blit(img, img.get_rect(center=center))

    def draw_start(self, surf):
        rr = road_rect()
        draw_text(surf, "FRAMERACING", (rr.centerx, HEIGHT//2 - 160), font=FONT_HUGE, color=(255, 255, 255), center=True)
        lines = [
            "Avoid cars, pass as many as you can.",
            "Power-ups: FAST, SLOW, BUBBLE, SHOOT",
            "SPACE to shoot when SHOOT is active",
            "←/→ lane, ↑/↓ up/down, P/ESC pause",
            "+20 per normal car, +50 special",
            "+1 life every 5000 points",
            "Press ENTER to start",
        ]
        for i, t in enumerate(lines):
            draw_text(surf, t, (rr.centerx, HEIGHT//2 - 60 + i*32), center=True)

    def draw_gameover(self, surf):
        import pygame, time
        rr = road_rect()
        draw_text(surf, "GAME OVER", (rr.centerx, HEIGHT//2 - 140), font=FONT_HUGE, color=(255, 255, 255), center=True)
        if self.name_allowed and time.time() < getattr(self, "top10_flash_until", 0):
            pygame.draw.rect(surf, (0,0,0,140), pygame.Rect(rr.centerx-160, HEIGHT//2 - 120, 320, 40))
            draw_text(surf, "NEW TOP-10!", (rr.centerx, HEIGHT//2 - 100), font=FONT_BIG, center=True)
        draw_text(surf, f"Score: {self.score}", (rr.centerx, HEIGHT//2 - 80), font=FONT_BIG, center=True)
        if self.name_allowed:
            draw_text(surf, "Enter your name:", (rr.centerx, HEIGHT//2 - 10), font=FONT_BIG, center=True)
            name_display = self.name_input if self.name_input else "PLAYER"
            pygame.draw.rect(surf, (240, 240, 240), pygame.Rect(rr.centerx - 150, HEIGHT//2 + 24, 300, 40), 2, border_radius=8)
            draw_text(surf, name_display, (rr.centerx, HEIGHT//2 + 30), center=True)
            draw_text(surf, "Press ENTER to save, ESC to restart", (rr.centerx, HEIGHT//2 + 90), center=True)
        else:
            draw_text(surf, "Not a Top 10 score", (rr.centerx, HEIGHT//2 - 10), font=FONT_BIG, center=True)
            draw_text(surf, "Press R to restart", (rr.centerx, HEIGHT//2 + 30), center=True)

    # Events
    def handle_event(self, event):
        global WIDTH, HEIGHT, screen
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = max(640, event.w), max(480, event.h)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            recalc_geometry(self.player, self.enemies, self.powerups)
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
                if self.state == "RUN":
                    self.state = "PAUSE"
                elif self.state == "PAUSE":
                    self.state = "RUN"
                elif self.state == "GAMEOVER" and self.name_allowed:
                    self.reset(full=False)
                else:
                    # Ignore ESC on START screen
                    pass
                return
            if self.state == "RUN":
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.player.move_lane(-1)
                    self.trail = []
                    self.trail_block_until = time.time() + 0.25
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.player.move_lane(+1)
                    self.trail = []
                    self.trail_block_until = time.time() + 0.25
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
                    if self.name_allowed:
                        name = self.name_input.strip() or "PLAYER"
                        save_score(name, self.score)
                    self.reset(full=False)
                elif self.name_allowed:
                    if event.key == pygame.K_ESCAPE:
                        self.reset(full=False)
                    elif event.key == pygame.K_BACKSPACE:
                        self.name_input = self.name_input[:-1]
                    else:
                        ch = event.unicode
                        if ch and (ch.isalnum() or ch in (" ", "_", "-")) and len(self.name_input) < 16:
                            self.name_input += ch
                elif event.key == pygame.K_r:
                    self.reset(full=False)

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    game = Game()
    dt = 0.0
    WIDTH, HEIGHT = screen.get_size()
    recalc_geometry(game.player, game.enemies, game.powerups)
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