import pygame
import random
import os
import json
import time
import base64
import io
import math
from datetime import datetime

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 640
TOPBAR_HEIGHT = 56

CELL_SIZE = 20
GRID_COLS = WINDOW_WIDTH // CELL_SIZE
GRID_ROWS = (WINDOW_HEIGHT - TOPBAR_HEIGHT) // CELL_SIZE

INITIAL_MOVE_DELAY_MS = 160
MIN_MOVE_DELAY_MS = 60
MAX_MOVE_DELAY_MS = 320
FOODS_PER_LEVEL = 5
POWERUP_MIN_INTERVAL = 6.0
POWERUP_MAX_INTERVAL = 12.0
POWERUP_LIFETIME = 10.0
SPEED_MOD_DURATION = 6.0
LEN_ICON_DURATION = 1.2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SAVE_DIR = os.path.join(ROOT_DIR, "presets", "setsave")
SAVE_FILE = os.path.join(SAVE_DIR, "framie_highscores.json")
BG_DIR = os.path.join(ROOT_DIR, "presets", "startup")
MAX_HISCORES = 10

WHITE  = (240, 240, 240)
BLACK  = (15, 15, 20)
GRAY   = (65, 70, 80)
DARK   = (35, 40, 50)
GREEN  = (60, 200, 85)
RED    = (220, 70, 70)
YELLOW = (240, 210, 70)
BLUE   = (70, 155, 240)
PURPLE = (160, 90, 210)
ORANGE = (245, 140, 60)
TEAL   = (70, 200, 200)

POP_WAV_B64 = "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"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class Button:
    def __init__(self, rect, label, callback, *, bg=GRAY, fg=WHITE, active_bg=None, get_active=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.bg = bg
        self.fg = fg
        self.active_bg = active_bg if active_bg else bg
        self.get_active = get_active

    def draw(self, surf, font, mouse_pos):
        is_hover = self.rect.collidepoint(mouse_pos)
        active = self.get_active() if self.get_active else False
        base = self.active_bg if active else self.bg
        color = tuple(clamp(c + (18 if is_hover else 0), 0, 255) for c in base)
        pygame.draw.rect(surf, color, self.rect, border_radius=10)
        pygame.draw.rect(surf, (35,40,50), self.rect, width=2, border_radius=10)
        label_surf = font.render(self.label, True, (240,240,240))
        surf.blit(label_surf, label_surf.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

class PowerUp:
    def __init__(self, pos, kind, born_time, lifetime=POWERUP_LIFETIME):
        self.pos = pos
        self.kind = kind
        self.born_time = born_time
        self.lifetime = lifetime
    def expired(self, now):
        return (now - self.born_time) > self.lifetime
    def color_and_text(self):
        if self.kind == 'len_plus': return (GREEN, '+')
        if self.kind == 'len_minus': return (ORANGE, '−')
        if self.kind == 'spd_plus': return (BLUE, '>>')
        if self.kind == 'spd_minus': return (PURPLE, '<<')
        if self.kind == 'skull': return (RED, 'SK')
        return (TEAL, '?')

class Effect:
    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.start = time.time()
        self.duration = kwargs.get('duration', 0.35 if kind == 'pulse' else 1.0 if kind == 'banner' else 0.25)
        self.kw = kwargs
    def alive(self):
        return (time.time() - self.start) < self.duration

class Particle:
    def __init__(self, x, y, vx, vy, color, life=0.6):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.born = time.time()
        self.color = color
    def alive(self):
        return (time.time() - self.born) < self.life
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= (1.0 - 1.5*dt)
        self.vy *= (1.0 - 1.5*dt)
    def draw(self, surf):
        age = time.time() - self.born
        t = clamp(age / self.life, 0.0, 1.0)
        alpha = int(255 * (1.0 - t))
        size = max(2, int(4 * (1.0 - t)))
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        s.fill((*self.color, alpha))
        surf.blit(s, (self.x - size//2, self.y - size//2))

class SnakeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Framie the Snake — Levels, Power-Ups, High Scores")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small  = pygame.font.SysFont("consolas,dejavusans,arial", 18)
        self.font_medium = pygame.font.SysFont("consolas,dejavusans,arial", 22, bold=True)
        self.font_large  = pygame.font.SysFont("consolas,dejavusans,arial", 32, bold=True)

        self.buttons = []
        self.snd_pop = None
        self.init_audio()

        self.reset(full=True)
        self.in_menu = True

        self.hiscores = self.load_highscores()

        self.bg_images = self.load_backgrounds()
        self.bg_index = -1
        self.bg_surface = None
        self.set_background_for_level(force=True)

        self.build_buttons()
        self.build_start_button()

    def init_audio(self):
        try:
            pygame.mixer.pre_init(22050, -16, 1, 512)
            pygame.mixer.init()
        except Exception:
            return
        try:
            assets_dir = os.path.join(ROOT_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)
            wav_path = os.path.join(assets_dir, "pop.wav")
            if not os.path.exists(wav_path):
                data = base64.b64decode(POP_WAV_B64.encode("ascii"))
                with open(wav_path, "wb") as f:
                    f.write(data)
            self.snd_pop = pygame.mixer.Sound(wav_path)
            self.snd_pop.set_volume(0.35)
        except Exception:
            self.snd_pop = None

    def play_pop(self):
        try:
            if self.snd_pop:
                self.snd_pop.play()
        except Exception:
            pass

    def ensure_save_dir(self):
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
        except Exception as e:
            print("Warning: could not create save dir:", e)

    def load_highscores(self):
        self.ensure_save_dir()
        old = os.path.join(SAVE_DIR, "snake_highscores.json")
        if os.path.exists(old) and not os.path.exists(SAVE_FILE):
            try: os.replace(old, SAVE_FILE)
            except Exception: pass
        # Migrate from helpers/presets/setsave -> root/presets/setsave if needed
        try:
            old_rel_dir = os.path.join(os.path.dirname(__file__), 'presets', 'setsave')
            for old_name in ('framie_highscores.json', 'snake_highscores.json'):
                old_path = os.path.join(old_rel_dir, old_name)
                if os.path.exists(old_path) and not os.path.exists(SAVE_FILE):
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    os.replace(old_path, SAVE_FILE)
        except Exception:
            pass
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[:MAX_HISCORES]
            except Exception as e:
                print("Failed to read highscores:", e)
        return []

    def save_highscores(self):
        self.ensure_save_dir()
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.hiscores[:MAX_HISCORES], f, indent=2)
        except Exception as e:
            print("Failed to save highscores:", e)

    def add_score(self):
        entry = {"score": int(self.score), "level": int(self.level), "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
        self.hiscores.append(entry)
        self.hiscores.sort(key=lambda x: (-x["score"], -x["level"]))
        self.hiscores = self.hiscores[:MAX_HISCORES]
        self.save_highscores()

    def load_backgrounds(self):
        paths = []
        try:
            if os.path.isdir(BG_DIR):
                exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                for root, _, files in os.walk(BG_DIR):
                    for name in files:
                        low = name.lower()
                        if low.startswith("logo_") and low.endswith(exts):
                            paths.append(os.path.join(root, name))
            paths.sort()
        except Exception as e:
            print("BG load error:", e)
        return paths

    def _scale_to_fill(self, image):
        iw, ih = image.get_size()
        sw, sh = WINDOW_WIDTH, WINDOW_HEIGHT
        if iw == 0 or ih == 0: return None
        scale = max(sw / iw, sh / ih)
        new_size = (int(iw * scale), int(ih * scale))
        img = pygame.transform.smoothscale(image, new_size)
        x = (new_size[0] - sw) // 2
        y = (new_size[1] - sh) // 2
        img = img.subsurface(pygame.Rect(x, y, sw, sh)).copy()
        img.set_alpha(int(255 * 0.75))
        return img

    def set_background_for_level(self, force=False):
        if not self.bg_images:
            self.bg_surface = None
            return
        if force:
            # Start on the *last uploaded* image (latest modified time)
            try:
                mtimes = [os.path.getmtime(p) for p in self.bg_images]
                self.bg_index = max(range(len(self.bg_images)), key=lambda i: mtimes[i])
            except Exception:
                self.bg_index = len(self.bg_images) - 1
        else:
            self.bg_index = (self.bg_index + 1) % len(self.bg_images)
        try:
            path = self.bg_images[self.bg_index]
            raw = pygame.image.load(path).convert()
            self.bg_surface = self._scale_to_fill(raw)
        except Exception as e:
            print("BG load/scale error:", e)
            self.bg_surface = None

    def reset(self, *, full=False):
        self.grid_cols = GRID_COLS
        self.grid_rows = GRID_ROWS
        start_x = self.grid_cols // 2
        start_y = self.grid_rows // 2
        self.snake = [(start_x, start_y), (start_x - 1, start_y)]
        self.direction = (1, 0)
        self.next_direction = self.direction
        self.pending_growth = 0
        self.score = 0
        self.level = 1
        self.foods_eaten = 0
        self.powerups = []
        self.food = None
        self.effects = []
        self.particles = []
        self.food = self.random_free_cell()
        self.base_move_delay = INITIAL_MOVE_DELAY_MS
        self.speed_mods = []
        self.move_delay = self.base_move_delay
        self.last_move_t = pygame.time.get_ticks()
        self.next_powerup_time = time.time() + random.uniform(POWERUP_MIN_INTERVAL, POWERUP_MAX_INTERVAL)
        self.paused = False
        self.show_help = False
        self.show_scores = False
        self.game_over = False
        self.game_over_added = False
        self.wrap_enabled = False
        self.status_icons = []
        if full: pygame.mouse.set_visible(True)

        # Bonus fruit state
        self.bonus_pos = None
        self.bonus_expires_at = 0.0
        self.bonus_spawned_level = 0
        # Spawn the first bonus fruit for level 1
        self.spawn_bonus_fruit()

    def spawn_bonus_fruit(self):
        """Spawn the flashing bonus fruit once per level for 10 seconds."""
        # Only once per level
        if getattr(self, 'bonus_spawned_level', 0) == self.level and self.bonus_pos:
            return
        # Pick a free cell
        tries = 0
        while True:
            pos = self.random_free_cell()
            if pos != getattr(self, 'food', None):
                break
            tries += 1
            if tries > 200:
                break
        self.bonus_pos = pos
        self.bonus_expires_at = time.time() + 10.0
        self.bonus_spawned_level = self.level

    def draw_bonus_fruit(self, surf):
        if not self.bonus_pos: return
        # Compute flashing color by cycling hue over time
        hue = (time.time() * 180) % 360  # degrees
        c = pygame.Color(0,0,0)
        try:
            c.hsva = (hue, 100, 100, 100)
        except Exception:
            c = pygame.Color(255, 255, 0)
        color = (c.r, c.g, c.b)
        x,y = self.bonus_pos
        px = x * CELL_SIZE
        py = TOPBAR_HEIGHT + y * CELL_SIZE
        # Draw as a circle-ish rect with small pulse
        pad = 2
        size = CELL_SIZE - pad*2
        pygame.draw.rect(surf, color, (px+pad, py+pad, size, size), border_radius=8)
        # Small white shine
        pygame.draw.rect(surf, (240,240,240), (px+pad+3, py+pad+3, 6, 6), border_radius=3)


    def build_buttons(self):
        pad = 10
        btn_h = TOPBAR_HEIGHT - pad*2
        y = pad
        x = pad
        def add_btn(label, callback, active_bg, get_active):
            nonlocal x
            text_w, _ = self.font_medium.size(label)
            w = max(140, text_w + 36)
            rect = (x, y, w, btn_h)
            self.buttons.append(Button(rect, label, callback, active_bg=active_bg, get_active=get_active))
            x += w + pad
        self.buttons = []
        add_btn("Pause", self.toggle_pause, (95,120,165), lambda: self.paused)
        add_btn("Exit", self.do_exit, (140, 60, 60), None)
        add_btn("Help", self.toggle_help, (95,165,120), lambda: self.show_help)
        add_btn("High Scores", self.toggle_scores, (165,120,95), lambda: self.show_scores)
        add_btn(f"Wrap: {'On' if self.wrap_enabled else 'Off'}", self.toggle_wrap, (120,120,95), lambda: self.wrap_enabled)

    def build_start_button(self):
        w, h = 280, 96
        cx = WINDOW_WIDTH//2 - w//2
        cy = TOPBAR_HEIGHT + 160
        self.start_button = Button((cx, cy, w, h), "Start", self.start_game, bg=(85,120,90), active_bg=(95,160,100))

    def in_bounds(self, cell):
        x,y = cell
        return 0 <= x < self.grid_cols and 0 <= y < self.grid_rows

    def random_free_cell(self):
        occupied = set(self.snake)
        if getattr(self, "food", None): occupied.add(self.food)
        if getattr(self, "powerups", None) is not None:
            occupied |= {pu.pos for pu in self.powerups}
        while True:
            x = random.randrange(0, self.grid_cols)
            y = random.randrange(0, self.grid_rows)
            if (x,y) not in occupied:
                return (x,y)

    def spawn_powerup(self):
        now = time.time()
        if now < self.next_powerup_time: return
        kinds = ['len_plus', 'len_minus', 'spd_plus', 'spd_minus']
        if self.level >= 4: kinds.append('skull')
        weights = []
        for k in kinds:
            if k == 'skull': weights.append(1.0)
            elif 'minus' in k: weights.append(1.2)
            else: weights.append(1.6)
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        kind = kinds[0]
        for k, w in zip(kinds, weights):
            if upto + w >= r:
                kind = k
                break
            upto += w
        pos = self.random_free_cell()
        self.powerups.append(PowerUp(pos, kind, now, POWERUP_LIFETIME))
        self.next_powerup_time = now + random.uniform(POWERUP_MIN_INTERVAL, POWERUP_MAX_INTERVAL)

    def _head_center_px(self):
        hx, hy = self.snake[0]
        return (hx * CELL_SIZE + CELL_SIZE//2, TOPBAR_HEIGHT + hy * CELL_SIZE + CELL_SIZE//2)

    def add_pulse(self, color, duration=0.35):
        cx, cy = self._head_center_px()
        self.effects.append(Effect('pulse', color=color, cx=cx, cy=cy, duration=duration))

    def add_banner(self, msg, color=YELLOW, duration=1.2):
        self.effects.append(Effect('banner', msg=msg, color=color, duration=duration))

    def add_flash(self, color=YELLOW, duration=0.25):
        self.effects.append(Effect('flash', color=color, duration=duration))

    def add_status_icon(self, kind, label, color, duration):
        self.status_icons.append({"kind": kind, "label": label, "color": color, "start": time.time(), "duration": duration})

    def apply_powerup(self, pu):
        if pu.kind == 'len_plus':
            self.pending_growth += 3
            self.add_pulse(GREEN); self.add_banner("LONGER!", GREEN, 0.6)
            self.add_status_icon('len_plus', '+', GREEN, LEN_ICON_DURATION)
        elif pu.kind == 'len_minus':
            for _ in range(3):
                if len(self.snake) > 2: self.snake.pop()
            self.add_pulse(ORANGE); self.add_banner("SHORTER!", ORANGE, 0.6)
            self.add_status_icon('len_minus', '−', ORANGE, LEN_ICON_DURATION)
        elif pu.kind == 'spd_plus':
            self.speed_mods.append({"delta": -20, "start": time.time(), "duration": SPEED_MOD_DURATION})
            self.add_pulse(BLUE); self.add_banner("SPEED UP!", BLUE, 0.6)
            self.add_status_icon('spd_plus', '>>', BLUE, SPEED_MOD_DURATION)
        elif pu.kind == 'spd_minus':
            self.speed_mods.append({"delta": +20, "start": time.time(), "duration": SPEED_MOD_DURATION})
            self.add_pulse(PURPLE); self.add_banner("SLOW DOWN", PURPLE, 0.6)
            self.add_status_icon('spd_minus', '<<', PURPLE, SPEED_MOD_DURATION)
        elif pu.kind == 'skull':
            self.add_flash(RED, 0.35)
            self.trigger_game_over()

    def level_up(self):
        self.level += 1
        self.base_move_delay = clamp(self.base_move_delay - 8, MIN_MOVE_DELAY_MS, MAX_MOVE_DELAY_MS)
        self.add_flash(YELLOW, 0.28); self.add_banner(f"LEVEL {self.level}!", YELLOW, 1.0)
        self.set_background_for_level(force=False)

        # Spawn the once-per-level bonus fruit
        self.spawn_bonus_fruit()

    def trigger_game_over(self):
        self.game_over = True
        if not self.game_over_added:
            self.add_score(); self.game_over_added = True

    def start_game(self):
        self.reset(); self.in_menu = False; self.set_background_for_level(force=True)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_help: self.show_help = False
                    elif self.show_scores: self.show_scores = False
                    elif self.game_over:
                        self.game_over = False; self.in_menu = True
                    elif not self.in_menu: self.toggle_pause()
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self.in_menu: self.start_game()
                    elif not self.show_help and not self.show_scores and not self.game_over:
                        self.toggle_pause()
                elif event.key == pygame.K_h: self.toggle_help()
                elif event.key == pygame.K_F1: self.toggle_scores()
                elif event.key == pygame.K_t: self.toggle_wrap()
                elif self.game_over and event.key == pygame.K_r:
                    self.reset(); self.in_menu = False
                else:
                    if not (self.in_menu or self.paused or self.show_help or self.show_scores or self.game_over):
                        if event.key in (pygame.K_UP, pygame.K_w): self.set_direction((0,-1))
                        elif event.key in (pygame.K_DOWN, pygame.K_s): self.set_direction((0,1))
                        elif event.key in (pygame.K_LEFT, pygame.K_a): self.set_direction((-1,0))
                        elif event.key in (pygame.K_RIGHT, pygame.K_d): self.set_direction((1,0))
            for b in self.buttons: b.handle_event(event)
            if self.in_menu: self.start_button.handle_event(event)

    def toggle_wrap(self):
        self.wrap_enabled = not self.wrap_enabled; self.build_buttons()

    def set_direction(self, d):
        if len(self.snake) > 1:
            cur = self.direction
            if (cur[0] == -d[0] and cur[0] != 0) or (cur[1] == -d[1] and cur[1] != 0): return
        self.next_direction = d

    def update(self):
        if self.in_menu or self.paused or self.show_help or self.show_scores or self.game_over: return
        dt = self.clock.get_time() / 1000.0
        now = time.time()
        self.speed_mods = [m for m in self.speed_mods if (now - m["start"]) < m["duration"]]
        total_delta = sum(m["delta"] for m in self.speed_mods)
        self.move_delay = clamp(self.base_move_delay + total_delta, MIN_MOVE_DELAY_MS, MAX_MOVE_DELAY_MS)
        now_ms = pygame.time.get_ticks()
        if now_ms - self.last_move_t >= self.move_delay:
            self.last_move_t = now_ms; self.direction = self.next_direction; self.step()
        self.spawn_powerup()
        self.powerups = [pu for pu in self.powerups if not pu.expired(now)]
        # Bonus fruit timeout
        if getattr(self, 'bonus_pos', None) and time.time() >= self.bonus_expires_at:
            self.bonus_pos = None
        self.effects = [e for e in self.effects if e.alive()]
        for p in self.particles: p.update(dt)
        self.particles = [p for p in self.particles if p.alive()]
        self.status_icons = [s for s in self.status_icons if (now - s["start"]) < s["duration"]]

    def step(self):
        head_x, head_y = self.snake[0]; dx, dy = self.direction
        nx, ny = head_x + dx, head_y + dy
        if self.wrap_enabled:
            nx %= self.grid_cols; ny %= self.grid_rows; new_head = (nx, ny)
        else:
            new_head = (nx, ny)
            if not self.in_bounds(new_head): self.trigger_game_over(); return
        if new_head in self.snake: self.trigger_game_over(); return
        self.snake.insert(0, new_head)
        # Bonus fruit?
        if getattr(self, 'bonus_pos', None) and new_head == self.bonus_pos:
            self.score += 300
            self.pending_growth += 1
            self.bonus_pos = None
            self.add_pulse(YELLOW)
            self.play_pop()
        if new_head == self.food:
            self.score += 10; self.foods_eaten += 1; self.pending_growth += 1; self.food = self.random_free_cell()
            self.play_pop()
            cx, cy = self._head_center_px()
            speed_factor = (MAX_MOVE_DELAY_MS - self.move_delay) / (MAX_MOVE_DELAY_MS - MIN_MOVE_DELAY_MS + 1e-6)
            count = int(10 + 15 * speed_factor)
            for _ in range(count):
                ang = random.uniform(0, 2*math.pi)
                mag = 60 + 200 * speed_factor + random.uniform(-20, 20)
                vx = mag * math.cos(ang); vy = mag * math.sin(ang)
                self.particles.append(Particle(cx, cy, vx, vy, YELLOW, life=0.5 + 0.3*random.random()))
            if self.foods_eaten % FOODS_PER_LEVEL == 0: self.level_up()
        hit_index = None
        for i, pu in enumerate(self.powerups):
            if pu.pos == new_head: hit_index = i; break
        if hit_index is not None:
            pu = self.powerups.pop(hit_index); self.apply_powerup(pu)
        if self.pending_growth > 0: self.pending_growth -= 1
        else: self.snake.pop()

    def draw_grid(self, surf):
        for c in range(self.grid_cols+1):
            x = c * CELL_SIZE; pygame.draw.line(surf, (40,45,55), (x, TOPBAR_HEIGHT), (x, WINDOW_HEIGHT))
        for r in range(self.grid_rows+1):
            y = TOPBAR_HEIGHT + r * CELL_SIZE; pygame.draw.line(surf, (40,45,55), (0, y), (WINDOW_WIDTH, y))

    def draw_topbar(self, surf):
        pygame.draw.rect(surf, (30,34,44), (0,0,WINDOW_WIDTH,TOPBAR_HEIGHT))
        mouse = pygame.mouse.get_pos()
        for b in self.buttons: b.draw(surf, self.font_medium, mouse)
        text = f"Score: {self.score}   Level: {self.level}   Speed: {int(1000/self.move_delay)} tps"
        ts = self.font_medium.render(text, True, WHITE)
        right_x = WINDOW_WIDTH - ts.get_width() - 12
        min_x = self.buttons[-1].rect.right + 16
        x = max(right_x, min_x)
        surf.blit(ts, (x, (TOPBAR_HEIGHT - ts.get_height())//2))

    def draw_snake(self, surf):
        for i, (x,y) in enumerate(self.snake):
            px = x * CELL_SIZE; py = TOPBAR_HEIGHT + y * CELL_SIZE
            if i == 0: self.draw_head(surf, px, py)
            else: pygame.draw.rect(surf, (80, 180, 100), (px+1, py+1, CELL_SIZE-2, CELL_SIZE-2), border_radius=6)

    def draw_head(self, surf, px, py):
        pygame.draw.rect(surf, (40, 160, 80), (px+1, py+1, CELL_SIZE-2, CELL_SIZE-2), border_radius=8)
        dx, dy = self.direction
        ex = 5 if dx >= 0 else CELL_SIZE-7
        ey_top = 6 if dy >= 0 else 4
        ey_bottom = CELL_SIZE-8 if dy <= 0 else CELL_SIZE-10
        eye1 = pygame.Rect(px + ex, py + ey_top, 4, 4)
        eye2 = pygame.Rect(px + ex, py + ey_bottom, 4, 4)
        pygame.draw.rect(surf, WHITE, eye1, border_radius=2); pygame.draw.rect(surf, WHITE, eye2, border_radius=2)
        pupil_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        pupil_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        for r in (eye1, eye2):
            pr = r.copy(); pr.x += pupil_dx; pr.y += pupil_dy
            pygame.draw.rect(surf, (10,10,10), pr, border_radius=2)

    def draw_food(self, surf):
        x,y = self.food; px = x * CELL_SIZE; py = TOPBAR_HEIGHT + y * CELL_SIZE
        pygame.draw.rect(surf, YELLOW, (px+2, py+2, CELL_SIZE-4, CELL_SIZE-4), border_radius=6)

    def draw_powerups(self, surf):
        for pu in self.powerups:
            color, txt = pu.color_and_text()
            x,y = pu.pos; px = x * CELL_SIZE; py = TOPBAR_HEIGHT + y * CELL_SIZE
            pygame.draw.rect(surf, color, (px+2, py+2, CELL_SIZE-4, CELL_SIZE-4), border_radius=6)
            label = self.font_small.render(txt, True, BLACK)
            surf.blit(label, label.get_rect(center=(px + CELL_SIZE//2, py + CELL_SIZE//2)))

    def draw_status_row(self, surf):
        now = time.time(); x = self.buttons[-1].rect.right + 16; y = TOPBAR_HEIGHT + 6
        size = 24; gap = 8
        for s in self.status_icons[:8]:
            t = (now - s["start"]) / s["duration"]; t = clamp(t, 0.0, 1.0)
            rect = pygame.Rect(x, y, size, size)
            pygame.draw.rect(surf, (25,28,36), rect, border_radius=6)
            pygame.draw.rect(surf, (85,90,110), rect, width=2, border_radius=6)
            end_angle = 2*math.pi * (1.0 - t); pygame.draw.arc(surf, s["color"], rect.inflate(10,10), 0, end_angle, 3)
            lbl = self.font_small.render(s["label"], True, WHITE); surf.blit(lbl, lbl.get_rect(center=rect.center))
            x += size + gap

    def draw_effects(self, surf):
        now = time.time()
        fx_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        for e in self.effects:
            t = (now - e.start) / e.duration; t = clamp(t, 0.0, 1.0)
            if e.kind == 'pulse':
                radius = int(CELL_SIZE * (1.0 + 2.0 * t)); alpha = int(180 * (1.0 - t))
                ring = pygame.Surface((radius*2+4, radius*2+4), pygame.SRCALPHA)
                pygame.draw.circle(ring, (*e.kw['color'], alpha), (radius+2, radius+2), radius, width=4)
                fx_surf.blit(ring, (e.kw['cx'] - radius - 2, e.kw['cy'] - radius - 2))
            elif e.kind == 'flash':
                alpha = int(120 * (1.0 - t)); overlay_color = (*e.kw.get('color', YELLOW), alpha)
                fx_surf.fill(overlay_color)
        surf.blit(fx_surf, (0,0))
        for e in self.effects:
            if e.kind == 'banner':
                t = (now - e.start) / e.duration; t = clamp(t, 0.0, 1.0)
                alpha = int(255 * (1.0 - abs(2*t - 1.0))); msg = e.kw.get('msg', ''); color = e.kw.get('color', YELLOW)
                text_surf = self.font_large.render(msg, True, color); pad = 12
                badge = pygame.Surface((text_surf.get_width()+pad*2, text_surf.get_height()+pad), pygame.SRCALPHA)
                pygame.draw.rect(badge, (20,25,30, int(alpha*0.65)), badge.get_rect(), border_radius=12)
                badge.blit(text_surf, (pad, pad//2)); badge.set_alpha(alpha)
                cx = WINDOW_WIDTH//2 - badge.get_width()//2; cy = TOPBAR_HEIGHT + 16; self.screen.blit(badge, (cx, cy))

    def draw_start_menu(self, surf):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,160)); surf.blit(overlay, (0,0))
        title = self.font_large.render("Framie the Snake", True, (240,240,240))
        surf.blit(title, title.get_rect(center=(WINDOW_WIDTH//2, TOPBAR_HEIGHT + 80)))
        sub = self.font_medium.render("Click Start (or press Enter/Space) to play.", True, (220,220,220))
        surf.blit(sub, sub.get_rect(center=(WINDOW_WIDTH//2, TOPBAR_HEIGHT + 120)))
        self.start_button.draw(surf, self.font_large, pygame.mouse.get_pos())

    def draw_help_overlay(self, surf):
        lines = [
            "HELP — Framie the Snake",
            "",
            "Move: Arrow Keys or WASD   •   Pause/Resume: Space/Start",
            "Help: H / button           •   High Scores: F1 / button",
            "Exit: button only          •   Restart after Game Over: R",
            "Toggle Wrap: T / button    •   ESC: close overlays / pause / close Game Over",
            "",
            "Goal: Eat the yellow food, grow, and survive.",
            f"Level up every {FOODS_PER_LEVEL} foods. Higher levels = faster & tougher.",
            "",
            "Power-Ups: [+] longer, [−] shorter, [>>] faster, [<<] slower, [SK] skull (L4+)",
            "Backgrounds: put images in presets/startup named logo_1.*, logo_2.*, ...",
        ]
        self.draw_overlay_box(surf, lines)

    def draw_scores_overlay(self, surf):
        lines = ["HIGH SCORES (Top 10)", ""]
        if not self.hiscores: lines.append("No scores yet. Go set some!")
        else:
            for i, s in enumerate(self.hiscores, 1):
                lines.append(f"{i:2d}. {s['score']:4d} pts  —  Lvl {s['level']}   {s['date']}")
        self.draw_overlay_box(surf, lines)

    def draw_gameover_overlay(self, surf):
        lines = ["GAME OVER", "", f"Score: {self.score}    Level: {self.level}", "", "Press R to restart   •   Press Esc to return to Start"]
        self.draw_overlay_box(surf, lines, title_color=RED)

    def draw_overlay_box(self, surf, lines, title_color=WHITE):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,140)); surf.blit(overlay, (0,0))
        box_w = int(WINDOW_WIDTH * 0.78); box_h = int(WINDOW_HEIGHT * 0.6)
        box_x = (WINDOW_WIDTH - box_w)//2; box_y = (WINDOW_HEIGHT - box_h)//2
        pygame.draw.rect(surf, (25,28,36), (box_x, box_y, box_w, box_h), border_radius=14)
        pygame.draw.rect(surf, (85,90,110), (box_x, box_y, box_w, box_h), width=2, border_radius=14)
        y = box_y + 20
        for idx, line in enumerate(lines):
            is_title = (idx == 0); font = self.font_large if is_title else self.font_medium
            color = title_color if is_title else WHITE
            ts = font.render(line, True, color); surf.blit(ts, (box_x + 20, y))
            y += ts.get_height() + (12 if is_title else 6)

    def toggle_pause(self):
        if self.game_over or self.in_menu: return
        self.paused = not self.paused
    def toggle_help(self):
        self.show_help = not self.show_help
        if self.show_help: self.show_scores = False
    def toggle_scores(self):
        self.show_scores = not self.show_scores
        if self.show_scores: self.show_help = False
    def do_exit(self):
        pygame.quit(); raise SystemExit

    def run(self):
        while True:
            self.clock.tick(60); self.handle_input(); self.update(); self.draw()
    def draw(self):
        if getattr(self, 'bg_surface', None): self.screen.blit(self.bg_surface, (0,0))
        else: self.screen.fill(BLACK)
        self.draw_grid(self.screen); self.draw_topbar(self.screen); self.draw_food(self.screen)
        self.draw_bonus_fruit(self.screen)
        self.draw_powerups(self.screen); self.draw_snake(self.screen)
        for p in self.particles: p.draw(self.screen)
        self.draw_status_row(self.screen); self.draw_effects(self.screen)
        if self.in_menu: self.draw_start_menu(self.screen)
        elif self.show_help: self.draw_help_overlay(self.screen)
        elif self.show_scores: self.draw_scores_overlay(self.screen)
        elif self.game_over: self.draw_gameover_overlay(self.screen)
        pygame.display.flip()

if __name__ == "__main__":
    try:
        SnakeGame().run()
    except SystemExit:
        pass