#!/usr/bin/env python3
# FrameBreaker - Pygame breakout/brick breaker
# Shooter power-up + color-coded power-ups (first-letter labels).
# Indentation hotfix in BackgroundRenderer.__init__.
#
# Layout:
#   - Script:            <root>/helpers/framebreaker.py
#   - Background logos:  <root>/preset/startup/             [fallback: <root>/presets/startup/]
#   - Brick thumbnails:  <root>/presets/setsave/thumbs/     (optional)
#   - High scores JSON:  <root>/presets/setsave/framesave.json
#
import os
import sys
import json
import random
import math
import pygame
from pygame import Rect

# ------------------------------
# Config
# ------------------------------
GAME_TITLE = "FrameBreaker"
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 60

# Gameplay
INITIAL_LIVES = 3
BALL_SPEED = 7.0
BALL_SPEEDUP_PER_LEVEL = 0.5
PADDLE_WIDTH, PADDLE_HEIGHT = 160, 18
PADDLE_SPEED = 12
BRICK_COLS = 12
BRICK_ROWS_BASE = 6  # grows with level
BRICK_MARGIN = 6     # spacing between bricks
BRICK_TOP_OFFSET = 110

# Power-up
POWERUP_DROP_CHANCE = 0.18  # 18% per brick
POWERUP_FALL_SPEED = 4
POWERUP_DURATION = 12.0  # seconds for timed ones
MAX_BALLS = 4

# Shooter
SHOOTER_FIRE_RATE = 7.0   # bullets per second while active
BULLET_SPEED = 14.0
BULLET_WIDTH, BULLET_HEIGHT = 4, 16

# UI
TOPBAR_HEIGHT = 64
FONT_NAME = None  # default

# ------------------------------
# Paths
# ------------------------------
def root_dir():
    helpers_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(helpers_dir)

ROOT_DIR = root_dir()

PATH_LOGOS_PRIMARY = os.path.join(ROOT_DIR, "preset", "startup")    # singular
PATH_LOGOS_FALLBACK = os.path.join(ROOT_DIR, "presets", "startup")  # plural
PATH_THUMBS = os.path.join(ROOT_DIR, "presets", "setsave", "thumbs")
PATH_SCORES_DIR = os.path.join(ROOT_DIR, "presets", "setsave")
PATH_SCORES_FILE = os.path.join(PATH_SCORES_DIR, "framesave.json")

# ------------------------------
# Helpers
# ------------------------------
def ensure_scores_dir():
    try:
        os.makedirs(PATH_SCORES_DIR, exist_ok=True)
    except Exception as e:
        print("[WARN] Could not create scores dir:", e)

def scan_images(path):
    if not os.path.isdir(path):
        return []
    imgs = []
    for name in os.listdir(path):
        lower = name.lower()
        if lower.endswith((".jpg", ".jpeg", ".png")):
            imgs.append(os.path.join(path, name))
    return imgs

def load_surfaces_from_files(files):
    out = []
    for fp in files:
        try:
            s = pygame.image.load(fp).convert_alpha()
            out.append(s)
        except Exception as e:
            print(f"[WARN] Failed loading {fp}: {e}")
    return out

def load_logo_surfaces():
    files = scan_images(PATH_LOGOS_PRIMARY)
    if not files:
        files = scan_images(PATH_LOGOS_FALLBACK)
    def num_key(p):
        base = os.path.splitext(os.path.basename(p))[0]
        parts = base.split("_")
        try:
            return int(parts[-1])
        except Exception:
            return 10**9
    files.sort(key=num_key)
    return load_surfaces_from_files(files)

def load_thumb_surfaces():
    files = scan_images(PATH_THUMBS)
    return load_surfaces_from_files(files)

def rounded_image(image: pygame.Surface, size, radius: int):
    w, h = size
    img = pygame.transform.smoothscale(image, (w, h)).convert_alpha()
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
    rounded = img.copy()
    rounded.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return rounded

def make_cover_background(logo: pygame.Surface, target_size, opacity=0.15):
    W, H = target_size
    bg = pygame.Surface((W, H), pygame.SRCALPHA)
    if logo is None:
        bg.fill((20, 20, 24))
        return bg
    lw, lh = logo.get_size()
    if lw <= 0 or lh <= 0:
        bg.fill((20, 20, 24))
        return bg
    scale = max(W / lw, H / lh)  # cover
    tw, th = max(1, int(lw * scale)), max(1, int(lh * scale))
    scaled = pygame.transform.smoothscale(logo, (tw, th)).convert_alpha()
    alpha = max(0, min(255, int(255 * float(opacity))))
    scaled.set_alpha(alpha)
    x = (W - tw) // 2
    y = (H - th) // 2
    bg.blit(scaled, (x, y))
    return bg

def draw_text(surface, text, x, y, font, color=(240,240,240), anchor="topleft"):
    r = font.render(text, True, color)
    rect = r.get_rect()
    setattr(rect, anchor, (x, y))
    surface.blit(r, rect)
    return rect

# ------------------------------
# UI Elements
# ------------------------------
class Button:
    def __init__(self, rect, text, font, callback, tooltip=None):
        self.rect = Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.tooltip = tooltip
        self.hover = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

    def draw(self, screen):
        bg = (40, 40, 46) if not self.hover else (60, 60, 70)
        pygame.draw.rect(screen, bg, self.rect, border_radius=10)
        pygame.draw.rect(screen, (90, 90, 100), self.rect, 2, border_radius=10)
        txt = self.font.render(self.text, True, (230, 230, 240))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

# ------------------------------
# Entities
# ------------------------------
class Paddle:
    def __init__(self, x, y):
        self.rect = Rect(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.rect.midtop = (x, y)
        self.speed = PADDLE_SPEED
        self.expand_timer = 0.0
        self.shrink_timer = 0.0
        self.sticky_timer = 0.0
        self.shooter_timer = 0.0  # NEW

    def update(self, dt, keys):
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.rect.x += self.speed
        self.rect.x = max(0, min(SCREEN_WIDTH - self.rect.width, self.rect.x))
        # timers
        if self.expand_timer > 0: self.expand_timer -= dt
        if self.shrink_timer > 0: self.shrink_timer -= dt
        if self.sticky_timer > 0: self.sticky_timer -= dt
        if self.shooter_timer > 0: self.shooter_timer -= dt
        base_w = PADDLE_WIDTH
        if self.expand_timer > 0:
            base_w = int(PADDLE_WIDTH * 1.4)
        if self.shrink_timer > 0:
            base_w = int(PADDLE_WIDTH * 0.7)
        if self.expand_timer > 0 and self.shrink_timer > 0:
            if self.expand_timer >= self.shrink_timer:
                base_w = int(PADDLE_WIDTH * 1.2)
            else:
                base_w = int(PADDLE_WIDTH * 0.85)
        cx = self.rect.centerx
        self.rect.width = max(80, min(280, base_w))
        self.rect.centerx = cx
        self.rect.y = SCREEN_HEIGHT - 80

    def draw(self, screen):
        pygame.draw.rect(screen, (220, 220, 235), self.rect, border_radius=10)
        pygame.draw.rect(screen, (90, 90, 100), self.rect, 2, border_radius=10)

class Ball:
    def __init__(self, x, y, speed=BALL_SPEED):
        self.pos = pygame.Vector2(x, y)
        angle = random.uniform(-0.35*math.pi, -0.65*math.pi)
        self.vel = pygame.Vector2(speed*math.cos(angle), speed*math.sin(angle))
        self.radius = 9
        self.stuck_to_paddle = False

    def update(self, dt):
        if self.stuck_to_paddle:
            return
        self.pos += self.vel
        if self.pos.x <= self.radius:
            self.pos.x = self.radius
            self.vel.x *= -1
        if self.pos.x >= SCREEN_WIDTH - self.radius:
            self.pos.x = SCREEN_WIDTH - self.radius
            self.vel.x *= -1
        if self.pos.y <= TOPBAR_HEIGHT + self.radius:
            self.pos.y = TOPBAR_HEIGHT + self.radius
            self.vel.y *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (int(self.pos.x), int(self.pos.y)), self.radius)
        pygame.draw.circle(screen, (80, 80, 90), (int(self.pos.x), int(self.pos.y)), self.radius, 2)

    def rect(self):
        r = Rect(0, 0, self.radius*2, self.radius*2)
        r.center = (int(self.pos.x), int(self.pos.y))
        return r

class Bullet:
    def __init__(self, x, y):
        self.rect = Rect(int(x - BULLET_WIDTH//2), int(y - BULLET_HEIGHT), BULLET_WIDTH, BULLET_HEIGHT)
        self.vy = -BULLET_SPEED

    def update(self, dt):
        self.rect.y += int(self.vy)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect, border_radius=2)
        pygame.draw.rect(screen, (60, 60, 70), self.rect, 1, border_radius=2)

class Brick:
    def __init__(self, rect: Rect, image: pygame.Surface, hp=1, score=50):
        self.rect = rect
        self.image = image
        self.alive = True
        self.hp = hp
        self.score = score

    def hit(self):
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
            return True
        return False

    def draw(self, screen):
        if not self.alive: return
        screen.blit(self.image, self.rect)
        pygame.draw.rect(screen, (30,30,36), self.rect, 2, border_radius=10)

# ------------------------------
# Power-ups
# ------------------------------
POWER_TYPES = ["EXPAND", "SHRINK", "SLOW", "FAST", "STICKY", "MULTI", "LIFE", "SHOOTER"]

POWER_COLORS = {
    "EXPAND":  (102, 187, 106),   # green
    "SHRINK":  (239, 83, 80),     # red
    "SLOW":    (66, 165, 245),    # blue
    "FAST":    (255, 202, 40),    # yellow
    "STICKY":  (171, 71, 188),    # purple
    "MULTI":   (255, 112, 67),    # orange
    "LIFE":    (236, 64, 122),    # pink
    "SHOOTER": (38, 166, 154),    # teal
}

class PowerUp:
    def __init__(self, kind, x, y):
        self.kind = kind
        self.rect = Rect(int(x)-16, int(y)-16, 32, 32)
        self.vy = POWERUP_FALL_SPEED

    def update(self, dt):
        self.rect.y += self.vy

    def draw(self, screen, font):
        color = POWER_COLORS.get(self.kind, (230,230,255))
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        pygame.draw.rect(screen, (30, 30, 40), self.rect, 2, border_radius=6)
        letter = self.kind[0] if self.kind else "?"
        txt = font.render(letter, True, (20,20,24))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

# ------------------------------
# Level + Background
# ------------------------------
class LevelManager:
    def __init__(self, logos, thumbs):
        self.logos = logos
        self.thumbs = thumbs
        self.level_index = 0

    def current_bg_logo(self):
        if not self.logos: return None
        idx = self.level_index % len(self.logos)
        return self.logos[idx]

    def brick_surface_for_row(self, row):
        src = self.thumbs if self.thumbs else self.logos
        if not src: return None
        idx = (self.level_index + row) % len(src)
        return src[idx]

    def make_bricks(self):
        rows = BRICK_ROWS_BASE + (self.level_index // 2)
        rows = min(rows, 10)
        total_margin = (BRICK_COLS + 1) * BRICK_MARGIN
        brick_w = (SCREEN_WIDTH - total_margin) // BRICK_COLS
        brick_h = 48
        bricks = []
        for r in range(rows):
            y = BRICK_TOP_OFFSET + r * (brick_h + BRICK_MARGIN)
            x_offset = BRICK_MARGIN if r % 2 == 0 else BRICK_MARGIN + brick_w//4
            cols = BRICK_COLS - (1 if r % 2 else 0)
            for c in range(cols):
                x = x_offset + c * (brick_w + BRICK_MARGIN)
                rect = Rect(x, y, brick_w, brick_h)
                surf = self.brick_surface_for_row(r)
                if surf is None:
                    img = pygame.Surface((brick_w, brick_h), pygame.SRCALPHA)
                    img.fill((200, 200, 220))
                else:
                    img = rounded_image(surf, (brick_w, brick_h), radius=12)
                hp = 1 + (self.level_index // 3)
                score = 50 + 10*self.level_index
                bricks.append(Brick(rect, img, hp=hp, score=score))
        return bricks

    def next_level(self):
        self.level_index += 1

class BackgroundRenderer:
    def __init__(self, level_manager):
        self.level_manager = level_manager
        self.cached_idx = -1
        self.cached_surface = None  # FIXED indent

    def get_surface(self):
        idx = self.level_manager.level_index
        if idx != self.cached_idx:
            logo = self.level_manager.current_bg_logo()
            self.cached_surface = make_cover_background(logo, (SCREEN_WIDTH, SCREEN_HEIGHT), opacity=0.15)
            self.cached_idx = idx
        return self.cached_surface

# ------------------------------
# High Scores
# ------------------------------
class HighScores:
    def __init__(self):
        ensure_scores_dir()
        self.file_path = PATH_SCORES_FILE
        self.scores = self.load_scores()

    def load_scores(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "scores" in data and isinstance(data["scores"], list):
                return data["scores"]
        except Exception:
            pass
        return []

    def add_score(self, score):
        self.scores.append({"score": int(score)})
        self.scores.sort(key=lambda s: s["score"], reverse=True)
        self.scores = self.scores[:10]
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.scores, f, indent=2)
        except Exception as e:
            print("[WARN] Could not save scores:", e)

# ------------------------------
# Game States
# ------------------------------
STATE_MENU = "MENU"
STATE_PLAYING = "PLAYING"
STATE_PAUSED = "PAUSED"
STATE_GAMEOVER = "GAMEOVER"
STATE_INFO = "INFO"

# ------------------------------
# Main Game
# ------------------------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(GAME_TITLE)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.Font(FONT_NAME, 22)
        self.font = pygame.font.Font(FONT_NAME, 28)
        self.font_big = pygame.font.Font(FONT_NAME, 46)

        self.state = STATE_MENU
        self.running = True

        self.logos = load_logo_surfaces()
        self.thumbs = load_thumb_surfaces()
        if not self.logos and not self.thumbs:
            placeholder = pygame.Surface((256, 256), pygame.SRCALPHA)
            placeholder.fill((120, 120, 160))
            pygame.draw.circle(placeholder, (255, 255, 255), (128, 128), 96, 8)
            self.logos = [placeholder]

        self.levels = LevelManager(self.logos, self.thumbs)
        self.background = BackgroundRenderer(self.levels)
        self.scores = HighScores()

        self.bullets = []
        self.shooter_cooldown = 0.0

        self.buttons = []
        self._build_topbar_buttons()

        self.menu_buttons = []
        self._build_menu_buttons()

        self.info_text = [
            "FrameBreaker — photo-brick breakout!",
            "Left/Right or A/D to move; Space/Click launches ball when sticky.",
            "Destroy all 'frame' bricks to advance.",
            "Power-ups (color + letter):",
            "  E Expand (green), S Shrink (red), S Slow (blue), F Fast (yellow),",
            "  S Sticky (purple), M Multi (orange), L Life (pink), S Shooter (teal)",
            "Shooter auto-fires while active; bullets break bricks.",
            "Logos from /preset/startup; thumbnails from /presets/setsave/thumbs.",
            "High scores saved to /presets/setsave/framesave.json",
        ]

        self.reset_game(full=True)

    def _build_topbar_buttons(self):
        pad = 10
        w, h = 120, 40
        x = SCREEN_WIDTH - (w + pad)
        y = (TOPBAR_HEIGHT - h) // 2

        def on_exit(): self.running = False
        def on_pause():
            if self.state == STATE_PLAYING:
                self.state = STATE_PAUSED
            elif self.state == STATE_PAUSED:
                self.state = STATE_PLAYING
        def on_info(): self.state = STATE_INFO

        self.buttons = [
            Button((x, y, w, h), "Exit", self.font, on_exit),
            Button((x - (w + pad), y, w, h), "Info", self.font, on_info),
            Button((x - 2*(w + pad), y, w, h), "Pause", self.font, on_pause),
        ]

    def _build_menu_buttons(self):
        bw, bh = 260, 60
        cx = SCREEN_WIDTH // 2
        top = SCREEN_HEIGHT//2 - 30
        spacing = 80

        def start():
            self.reset_game(full=True)
            self.state = STATE_PLAYING
        def info():
            self.state = STATE_INFO
        def quit_():
            self.running = False

        self.menu_buttons = [
            Button((cx - bw//2, top, bw, bh), "Start Game", self.font_big, start),
            Button((cx - bw//2, top + spacing, bw, bh), "Info", self.font, info),
            Button((cx - bw//2, top + 2*spacing, bw, bh), "Exit", self.font, quit_),
        ]

    def reset_game(self, full=False):
        self.lives = INITIAL_LIVES
        self.score = 0
        self.levels.level_index = 0
        self._start_level(reset_balls=True)

    def _start_level(self, reset_balls=False):
        self.bricks = self.levels.make_bricks()
        self.powerups = []
        self.bullets.clear()
        self.shooter_cooldown = 0.0
        self.balls = [Ball(SCREEN_WIDTH//2, SCREEN_HEIGHT//2, speed=BALL_SPEED + self.levels.level_index*BALL_SPEEDUP_PER_LEVEL)]
        self.paddle = Paddle(SCREEN_WIDTH//2, SCREEN_HEIGHT - 90)
        if reset_balls:
            for b in self.balls:
                b.stuck_to_paddle = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if self.state in (STATE_MENU, STATE_INFO, STATE_PAUSED, STATE_PLAYING, STATE_GAMEOVER):
                if self.state != STATE_MENU:
                    for btn in self.buttons: btn.handle_event(event)
            if self.state == STATE_MENU:
                for btn in self.menu_buttons: btn.handle_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_p):
                    if self.state == STATE_PLAYING:
                        self.state = STATE_PAUSED
                    elif self.state == STATE_PAUSED:
                        self.state = STATE_PLAYING
                    elif self.state in (STATE_INFO, STATE_GAMEOVER):
                        self.state = STATE_MENU

    def update(self, dt):
        keys = pygame.key.get_pressed()
        if self.state == STATE_PLAYING:
            self.paddle.update(dt, keys)

            if self.paddle.shooter_timer > 0:
                self.shooter_cooldown -= dt
                if self.shooter_cooldown <= 0.0:
                    left_x = self.paddle.rect.left + self.paddle.rect.width * 0.3
                    right_x = self.paddle.rect.left + self.paddle.rect.width * 0.7
                    y = self.paddle.rect.top
                    self.bullets.append(Bullet(left_x, y))
                    self.bullets.append(Bullet(right_x, y))
                    self.shooter_cooldown = 1.0 / SHOOTER_FIRE_RATE

            for ball in list(self.balls):
                ball.update(dt)
                if ball.rect().colliderect(self.paddle.rect) and ball.vel.y > 0:
                    offset = (ball.pos.x - self.paddle.rect.centerx) / (self.paddle.rect.width/2)
                    angle = -math.pi/2 * 0.75 * offset
                    speed = ball.vel.length()
                    speed = max(5.5, min(speed, 12.0))
                    ball.vel.y = -abs(speed * math.cos(angle))
                    ball.vel.x = speed * math.sin(angle)
                    if self.paddle.sticky_timer > 0:
                        ball.stuck_to_paddle = True

                if ball.stuck_to_paddle:
                    ball.pos.x = max(self.paddle.rect.left+ball.radius, min(self.paddle.rect.right-ball.radius, pygame.mouse.get_pos()[0]))
                    ball.pos.y = self.paddle.rect.top - ball.radius - 1
                    if keys[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]:
                        ball.stuck_to_paddle = False
                        if ball.vel.length() == 0:
                            ball.vel = pygame.Vector2(0, -BALL_SPEED)

                if ball.pos.y > SCREEN_HEIGHT + ball.radius:
                    self.balls.remove(ball)
                    if not self.balls:
                        self.lives -= 1
                        if self.lives <= 0:
                            self.scores.add_score(self.score)
                            self.state = STATE_GAMEOVER
                        else:
                            self.balls = [Ball(self.paddle.rect.centerx, self.paddle.rect.top - 20, speed=BALL_SPEED + self.levels.level_index*BALL_SPEEDUP_PER_LEVEL)]
                            self.balls[0].stuck_to_paddle = True

            for ball in self.balls:
                ball_rect = ball.rect()
                for brick in self.bricks:
                    if not brick.alive: continue
                    if ball_rect.colliderect(brick.rect):
                        overlap_left = ball_rect.right - brick.rect.left
                        overlap_right = brick.rect.right - ball_rect.left
                        overlap_top = ball_rect.bottom - brick.rect.top
                        overlap_bottom = brick.rect.bottom - ball_rect.top
                        min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
                        if min_overlap == overlap_left:
                            ball.pos.x -= overlap_left
                            ball.vel.x *= -1
                        elif min_overlap == overlap_right:
                            ball.pos.x += overlap_right
                            ball.vel.x *= -1
                        elif min_overlap == overlap_top:
                            ball.pos.y -= overlap_top
                            ball.vel.y *= -1
                        else:
                            ball.pos.y += overlap_bottom
                            ball.vel.y *= -1

                        destroyed = brick.hit()
                        if destroyed:
                            self.score += brick.score
                            if random.random() < POWERUP_DROP_CHANCE:
                                kind = random.choice(POWER_TYPES)
                                self.powerups.append(PowerUp(kind, brick.rect.centerx, brick.rect.centery))
                        break

            for bullet in list(self.bullets):
                bullet.update(dt)
                if bullet.rect.bottom < TOPBAR_HEIGHT:
                    self.bullets.remove(bullet)
                    continue
                hit_something = False
                for brick in self.bricks:
                    if not brick.alive: continue
                    if bullet.rect.colliderect(brick.rect):
                        hit_something = True
                        destroyed = brick.hit()
                        if destroyed:
                            self.score += brick.score
                            if random.random() < POWERUP_DROP_CHANCE:
                                kind = random.choice(POWER_TYPES)
                                self.powerups.append(PowerUp(kind, brick.rect.centerx, brick.rect.centery))
                        break
                if hit_something and bullet in self.bullets:
                    self.bullets.remove(bullet)

            for p in list(self.powerups):
                p.update(dt)
                if p.rect.top > SCREEN_HEIGHT:
                    self.powerups.remove(p)
                elif p.rect.colliderect(self.paddle.rect):
                    self.apply_powerup(p.kind)
                    self.powerups.remove(p)

            if all((not b.alive) for b in self.bricks):
                self.levels.next_level()
                self._start_level(reset_balls=False)
                for b in self.balls:
                    b.stuck_to_paddle = True

    def apply_powerup(self, kind):
        if kind == "EXPAND":
            self.paddle.expand_timer = POWERUP_DURATION
        elif kind == "SHRINK":
            self.paddle.shrink_timer = POWERUP_DURATION
        elif kind == "SLOW":
            for b in self.balls:
                b.vel *= 0.8
        elif kind == "FAST":
            for b in self.balls:
                b.vel *= 1.2
        elif kind == "STICKY":
            self.paddle.sticky_timer = POWERUP_DURATION
        elif kind == "MULTI":
            additions = min(MAX_BALLS - len(self.balls), 2)
            for _ in range(additions):
                if not self.balls: break
                base = random.choice(self.balls)
                nb = Ball(base.pos.x, base.pos.y, speed=base.vel.length())
                angle = random.uniform(-math.pi/3, -2*math.pi/3)
                speed = base.vel.length()
                nb.vel = pygame.Vector2(speed*math.cos(angle), speed*math.sin(angle))
                self.balls.append(nb)
        elif kind == "LIFE":
            self.lives += 1
        elif kind == "SHOOTER":
            self.paddle.shooter_timer = POWERUP_DURATION
            self.shooter_cooldown = 0.0

    def draw_topbar(self):
        bar = Rect(0, 0, SCREEN_WIDTH, TOPBAR_HEIGHT)
        pygame.draw.rect(self.screen, (18, 18, 22), bar)
        pygame.draw.line(self.screen, (60, 60, 70), (0, TOPBAR_HEIGHT), (SCREEN_WIDTH, TOPBAR_HEIGHT), 2)
        draw_text(self.screen, f"Score: {self.score}", 20, TOPBAR_HEIGHT//2, self.font, anchor="midleft")
        draw_text(self.screen, f"Lives: {self.lives}", 250, TOPBAR_HEIGHT//2, self.font, anchor="midleft")
        draw_text(self.screen, f"Level: {self.levels.level_index + 1}", 420, TOPBAR_HEIGHT//2, self.font, anchor="midleft")
        if self.paddle.shooter_timer > 0:
            draw_text(self.screen, "Shooter", 580, TOPBAR_HEIGHT//2, self.font_small, (160, 220, 215), anchor="midleft")
        for btn in self.buttons:
            btn.draw(self.screen)

    def draw(self):
        self.screen.blit(self.background.get_surface(), (0, 0))

        if self.state == STATE_MENU:
            title_rect = draw_text(self.screen, "FrameBreaker", SCREEN_WIDTH//2, 150, self.font_big, anchor="midtop")
            subtitle = "Brick breaker with photo frames"
            draw_text(self.screen, subtitle, SCREEN_WIDTH//2, title_rect.bottom + 10, self.font, (210,210,220), anchor="midtop")
            draw_text(self.screen, "High Scores", SCREEN_WIDTH//2, title_rect.bottom + 70, self.font, anchor="midtop")
            for i, s in enumerate(self.scores.scores[:8]):
                draw_text(self.screen, f"{i+1:2d}. {s['score']}", SCREEN_WIDTH//2, title_rect.bottom + 110 + i*28, self.font_small, anchor="midtop")
            for btn in self.menu_buttons:
                btn.draw(self.screen)

        elif self.state in (STATE_PLAYING, STATE_PAUSED):
            for b in self.bricks:
                b.draw(self.screen)
            self.paddle.draw(self.screen)
            for ball in self.balls:
                ball.draw(self.screen)
            for p in self.powerups:
                p.draw(self.screen, self.font)
            for bullet in self.bullets:
                bullet.draw(self.screen)

            if self.state == STATE_PAUSED:
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                self.screen.blit(overlay, (0,0))
                draw_text(self.screen, "Paused", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40, self.font_big, anchor="midtop")
                draw_text(self.screen, "Press P or Esc to resume", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 10, self.font, anchor="midtop")

        elif self.state == STATE_INFO:
            y = 140
            draw_text(self.screen, "How to Play", SCREEN_WIDTH//2, 90, self.font_big, anchor="midtop")
            for line in self.info_text:
                draw_text(self.screen, line, SCREEN_WIDTH//2, y, self.font, (230,230,240), anchor="midtop")  # keep simple
                y += 34
            draw_text(self.screen, "Press Esc to return", SCREEN_WIDTH//2, y+20, self.font, (230,230,240), anchor="midtop")

        elif self.state == STATE_GAMEOVER:
            draw_text(self.screen, "Game Over", SCREEN_WIDTH//2, 160, self.font_big, anchor="midtop")
            draw_text(self.screen, f"Final Score: {self.score}", SCREEN_WIDTH//2, 230, self.font, anchor="midtop")
            draw_text(self.screen, "Press Esc to return to menu", SCREEN_WIDTH//2, 280, self.font, anchor="midtop")

        self.draw_topbar()
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
