# -*- coding: utf-8 -*-
"""
Tetris Easter Egg — helpers/tetris_game.py

New in this build:
- **Left Info Panel**: full Top 10 highscores + keyboard shortcuts moved to a dedicated left column.
- **Clickable Save/Skip**: when entering a highscore name, you now get visible buttons.
- Keeps: space to rotate, ghost piece, next preview, Start/Pause/Reset/Exit, DAS/ARR hold-to-move, celebrations.

Controls:
  ←/→ move (auto-repeat) • ↓ soft drop • SPACE/↑ rotate • P pause • R reset • Esc exit
"""
import os, sys, math, json, time, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import warnings
warnings.filterwarnings(
    'ignore',
    message=r'pkg_resources is deprecated as an API',
    category=UserWarning,
    module=r'pygame\.pkgdata'
)
try:
    import pygame
except Exception:
    print("This module requires pygame. Install with: pip install pygame")
    raise



def _presets_scores_path():
    """Return the highscores file path under <app root>/presets/setsave/."""
    from pathlib import Path
    root = Path(__file__).resolve().parent
    # if running from .../helpers/, go up one level to app root
    if root.name.lower() == "helpers":
        root = root.parent
    save_dir = root / "presets" / "setsave"
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return str(save_dir / "tetris_scores.json")

def _backgrounds_dir() -> Path:
    """Return the backgrounds directory under <app root>/presets/startup/."""
    root = Path(__file__).resolve().parent
    if root.name.lower() == "helpers":
        root = root.parent
    return root / "presets" / "startup"

def _list_bg_images():
    """Return sorted list of background image file paths (logo_#.jpg/.jpeg/.png)."""
    p = _backgrounds_dir()
    files = []
    try:
        for f in p.iterdir():
            name = f.name.lower()
            if name.startswith("logo_") and (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")):
                import re
                m = re.search(r'logo_(\d+)', name)
                n = int(m.group(1)) if m else 0
                files.append((n, f))
        files.sort(key=lambda t: t[0])
    except Exception:
        pass
    return [str(f) for _, f in files]


# ------------------------------- Config -------------------------------
GRID_W = 10
GRID_H = 20
CELL = 30               # pixels per cell
LEFTBAR_W = 240         # NEW: left info panel width
SIDEBAR_W = 240         # right sidebar width
MARGIN = 12
FPS = 60

# --- Effects & feel ---
EFFECT_MS = 1000             # existing global effects
LINE_CLEAR_MS = 280          # per-row flash/fade duration (ms)
SPARK_MS = 360               # land/rotate sparks lifetime (ms)
SHAKE_MS = 240               # tetris shake (ms)
SHAKE_MAG = 6                # max pixels of shake

# Colors
BLACK   = (15, 15, 18)
DARK    = (28, 28, 36)
LIGHT   = (230, 230, 240)
DIM     = (180, 180, 190)
ACCENT  = (120, 180, 255)
RED     = (230, 70, 70)
GREEN   = (70, 200, 120)
YELLOW  = (240, 210, 80)
MAGENTA = (210, 120, 220)
CYAN    = (70, 220, 220)
ORANGE  = (250, 160, 70)
BLUE    = (85, 140, 255)
WHITE   = (255, 255, 255)

# Tetromino colors in I, J, L, O, S, T, Z order
PIECE_COLORS = [CYAN, BLUE, ORANGE, YELLOW, GREEN, MAGENTA, RED]

# Scoring
LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}

# Falling speeds (ms)
BASE_DROP_MS = 800
LEVEL_MS_STEP = 60
MIN_DROP_MS = 120
SOFT_DROP_MS = 40

# Auto-shift (hold left/right)
DAS_MS = 170   # delay before the first repeat
ARR_MS = 45    # repeat interval while held

# Effects
EFFECT_MS = 1000

FONT_NAME = None  # default system font

def _default_scores_path() -> Path:
    app_name = os.getenv("FRAMEVISION_APP_NAME", "FrameVision")
    if os.name == "nt":
        lad = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if lad:
            d = Path(lad) / app_name
            d.mkdir(parents=True, exist_ok=True)
            return d / "tetris_scores.json"
    return Path.home() / f".{app_name.lower()}_tetris_scores.json"


# --------------------------- Tetromino data ---------------------------
PIECES = {
    'I': [[1,1,1,1]],
    'J': [[1,0,0],[1,1,1]],
    'L': [[0,0,1],[1,1,1]],
    'O': [[1,1],[1,1]],
    'S': [[0,1,1],[1,1,0]],
    'T': [[0,1,0],[1,1,1]],
    'Z': [[1,1,0],[0,1,1]],
}
ORDER = ['I','J','L','O','S','T','Z']

def rotate_matrix(m):
    return [list(row) for row in zip(*m[::-1])]

def gen_rots(base):
    rots = [base]
    for _ in range(3):
        rots.append(rotate_matrix(rots[-1]))
    # dedupe (O piece, etc.)
    uniq, seen = [], set()
    for r in rots:
        key = tuple(tuple(row) for row in r)
        if key not in seen:
            uniq.append(r); seen.add(key)
    return uniq

ROTS = {k: gen_rots(v) for k,v in PIECES.items()}

@dataclass
class Tetromino:
    name: str
    ri: int
    x: int
    y: int
    @property
    def shape(self): return ROTS[self.name][self.ri]
    def rotate(self, d=1): self.ri = (self.ri + d) % len(ROTS[self.name])

class Bag7:
    def __init__(self): self.bag = []
    def next(self):
        if not self.bag:
            self.bag = ORDER[:]; random.shuffle(self.bag)
        return self.bag.pop()


# ------------------------------ Highscores ----------------------------
class Highscores:
    def __init__(self, path: Optional[Path]=None, limit=10):
        self.limit = limit
        self.path = path or _default_scores_path()
        self.scores = self._load()
    def _load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list): return data[:self.limit]
        except Exception: pass
        return []
    def save(self):
        try: self.path.write_text(json.dumps(self.scores[:self.limit], indent=2), encoding="utf-8")
        except Exception: pass
    def qualifies(self, score:int)->bool:
        return len(self.scores)<self.limit or (self.scores and score>self.scores[-1]["score"])
    def add(self, name:str, score:int):
        self.scores.append({"name":name, "score":int(score), "ts":int(time.time())})
        self.scores.sort(key=lambda s:s["score"], reverse=True)
        self.scores = self.scores[:self.limit]
        self.save()


# ------------------------------- UI bits ------------------------------
class Button:
    def __init__(self, r: pygame.Rect, text: str):
        self.rect = r
        self.text = text
        self.hover = False
    def draw(self, surf, font):
        pygame.draw.rect(surf, (70,70,80) if not self.hover else DIM, self.rect, border_radius=10)
        label = font.render(self.text, True, WHITE)
        surf.blit(label, label.get_rect(center=self.rect.center))


class Effect:
    def __init__(self, kind: str, started_ms: int, **data):
        # kinds: 'smiley','confetti','flash','sunglasses','sparkle','line_clear','land_sparks','rotate_sparks'
        self.kind = kind
        self.started_ms = started_ms
        self.data = data
        # allow per-effect duration override
        self.duration_ms = data.get("duration_ms", EFFECT_MS)

        # Particles: list of dicts with x,y,vx,vy,life_ms
        # For convenience, store coords relative to playfield origin (0,0).
        particles = []
        if self.kind == 'confetti':
            import random, math
            for _ in range(60):
                x = random.randint(0, GRID_W*CELL)
                y = random.randint(-80, -10)
                vx = random.uniform(-0.6, 0.6)
                vy = random.uniform(1.5, 3.2)
                r = random.randint(2, 4)
                particles.append({"x":x,"y":y,"vx":vx,"vy":vy,"r":r})
        elif self.kind == 'sparkle':
            import random, math
            for _ in range(40):
                x = random.randint(0, GRID_W*CELL)
                y = random.randint(0, GRID_H*CELL)
                phase = random.uniform(0, 2*math.pi)
                size = random.randint(2,4)
                particles.append({"x":x,"y":y,"phase":phase,"size":size})
        elif self.kind in ('land_sparks','rotate_sparks'):
            particles = data.get("particles", [])

        self.particles = particles

    def alive(self, now):
        return (now - self.started_ms) < self.duration_ms

    def draw(self, surf, now, fonts):
        import pygame, math, random
        elapsed = now - self.started_ms
        remain = max(0, self.duration_ms - elapsed)
        alpha = int(255 * (remain / max(1, self.duration_ms)))

        # Precompute playfield offset
        play_x = LEFTBAR_W + MARGIN
        play_y = MARGIN

        if self.kind == 'flash':
            overlay = pygame.Surface((GRID_W*CELL, GRID_H*CELL), pygame.SRCALPHA)
            overlay.fill((255,255,255,min(alpha, 180)))
            surf.blit(overlay,(0,0))  # legacy global flash
        elif self.kind == 'sunglasses':
            txt = fonts['big'].render("😎", True, WHITE)
            surf.blit(txt, txt.get_rect(center=(GRID_W*CELL//2, GRID_H*CELL//2)))
        elif self.kind == 'smiley':
            cx, cy = GRID_W*CELL//2, GRID_H*CELL//2
            radius = int(80 + 10*math.sin(now/90))
            pygame.draw.circle(surf, YELLOW, (cx,cy), radius)
            pygame.draw.circle(surf, BLACK, (cx - radius//3, cy - radius//3), radius//8)
            pygame.draw.circle(surf, BLACK, (cx + radius//3, cy - radius//3), radius//8)
            rect = pygame.Rect(cx - radius//2, cy - radius//8, radius, radius)
            pygame.draw.arc(surf, BLACK, rect, math.pi + math.pi/8, 2*math.pi - math.pi/8, 5)
        elif self.kind == 'confetti':
            for p in self.particles:
                p["x"] += p["vx"]; p["y"] += p["vy"]
                pygame.draw.circle(surf, (random.randint(150,255), random.randint(100,255), random.randint(100,255)), (int(p["x"]), int(p["y"])), p["r"])
        elif self.kind == 'sparkle':
            t = pygame.time.get_ticks()
            for p in self.particles:
                tw = (math.sin(t/120 + p["phase"]) + 1.2) * 0.5
                s = max(2, int(p["size"]*tw))
                x, y = int(p["x"]), int(p["y"])
                pygame.draw.line(surf, WHITE, (x - s, y), (x + s, y))
                pygame.draw.line(surf, WHITE, (x, y - s), (x, y + s))
                pygame.draw.line(surf, WHITE, (x - s, y - s), (x + s, y + s))
                pygame.draw.line(surf, WHITE, (x - s, y + s), (x + s, y - s))
        elif self.kind == 'line_clear':
            # Fade each full row on the playfield area
            rows = self.data.get("rows", [])
            # Flash stronger at start, then fade
            phase = elapsed / max(1, self.duration_ms)
            intensity = 200 if phase < 0.25 else int(200 * (1 - phase))
            intensity = max(0, min(200, intensity))
            for y in rows:
                r = pygame.Rect(play_x, play_y + y*CELL, GRID_W*CELL, CELL-1)
                overlay = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                overlay.fill((255,255,255,intensity))
                surf.blit(overlay, r.topleft)
        elif self.kind in ('land_sparks','rotate_sparks'):
            # Update and draw tiny particles
            for p in list(self.particles):
                life_ms = p.get("life_ms", SPARK_MS)
                age = elapsed - p.get("born_ms", 0)
                if age > life_ms:
                    self.particles.remove(p); continue
                p["x"] += p.get("vx",0); p["y"] += p.get("vy",0)
                # small decay
                p["vx"] *= 0.98; p["vy"] *= 0.98
                # fade by age
                a = max(0, 255 - int(255 * (age / life_ms)))
                rad = max(1, int(p.get("r",2)))
                px = int(play_x + p["x"]); py = int(play_y + p["y"])
                s = pygame.Surface((rad*2,rad*2), pygame.SRCALPHA)
                s.fill((255,255,255,a))
                surf.blit(s, (px-rad, py-rad))


# ------------------------------- Game core ----------------------------
class TetrisGame:
    def __init__(self, surface, fonts, scores:Highscores):
        self.surface = surface
        self.fonts = fonts
        self.scores = scores

        self.grid = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.bag = Bag7()
        self.current = self._spawn()
        self.next_name = self.bag.next()

        self.level = 0
        self.lines_total = 0
        self.score = 0
        self.drop_ms = BASE_DROP_MS
        self.last_drop = pygame.time.get_ticks()
        self.soft_drop = False

        self.started = False
        self.paused = False
        self.game_over = False

        self.effects: list[Effect] = []
        self.name_entry_active = False
        self.name_buffer = ""
        # Guard: ensure we only submit once per Game Over
        self.highscore_submitted = False

        # Buttons (rects will be positioned by _layout_sidebar())
        x0 = LEFTBAR_W + GRID_W*CELL + MARGIN*2
        bw = SIDEBAR_W - 2*MARGIN
        bh, gap = 42, 14
        base_y = MARGIN + 60
        self.buttons = {
            'Start': Button(pygame.Rect(x0, base_y, bw, bh), "Start"),
            'Pause': Button(pygame.Rect(x0, base_y + (bh+gap), bw, bh), "Pause"),
            'Reset': Button(pygame.Rect(x0, base_y + 2*(bh+gap), bw, bh), "Reset"),
            'Exit' : Button(pygame.Rect(x0, base_y + 3*(bh+gap), bw, bh), "Exit"),
        }

        # Name-entry overlay button rects (computed at draw time)
        self.overlay_save_rect = None
        self.overlay_skip_rect = None

        # Movement hold state
        self.key_left = False
        self.key_right = False
        self.move_dir = 0        # -1, 0, +1
        self.move_repeat_t = 0

        # Clear animation state
        self.clearing_rows = []
        self.clearing_started = 0
        self.clearing_score_pending = 0

        # Motion/accessibility
        self.reduce_motion = False
        self.shake_until = 0
        self.shake_mag = 0

        # --- Backgrounds (25% opacity, cycle each level) ---
        self.bg_paths = _list_bg_images()
        self.bg_images = []
        self.bg_index = 0
        self._bg_scaled = None
        self._bg_cache_key = None
        # Preload originals (no scaling yet)
        for pth in self.bg_paths:
            try:
                img = pygame.image.load(pth).convert()
                self.bg_images.append(img)
            except Exception:
                pass


    # ---- helpers ----
    
    # ---- background helpers ----
    def _advance_background(self):
        if getattr(self, "bg_images", None):
            self.bg_index = (self.bg_index + 1) % len(self.bg_images)
            self._bg_cache_key = None  # force rescale

    def _draw_background(self):
        # Draw current background scaled to fill the entire window at ~25% opacity.
        if not getattr(self, "bg_images", None):
            return
        try:
            W, H = self.surface.get_size()
            idx = self.bg_index % len(self.bg_images)
            key = (idx, W, H)
            if getattr(self, "_bg_cache_key", None) != key or self._bg_scaled is None:
                src = self.bg_images[idx]
                iw, ih = src.get_size()
                # cover: scale to fill while preserving aspect
                scale = max(W / max(1, iw), H / max(1, ih))
                sw, sh = max(1, int(iw * scale)), max(1, int(ih * scale))
                scaled = pygame.transform.smoothscale(src, (sw, sh))
                scaled.set_alpha(32)  # ~12% opacity
                self._bg_scaled = scaled
                self._bg_cache_key = key
            x = (W - self._bg_scaled.get_width()) // 2
            y = (H - self._bg_scaled.get_height()) // 2
            self.surface.blit(self._bg_scaled, (x, y))
        except Exception:
            # Backgrounds are optional; ignore failures
            pass

    def _on_level_change(self, new_level:int):
        if new_level != self.level:
            self.level = new_level
            self.drop_ms = max(MIN_DROP_MS, BASE_DROP_MS - self.level * LEVEL_MS_STEP)
            # Next background each time a new level starts
            self._advance_background()

    def _spawn(self) -> Tetromino:
        n = self.bag.next()
        shape0 = ROTS[n][0]
        x = (GRID_W - len(shape0[0])) // 2
        return Tetromino(n, 0, x, -2)
    def _spawn_land_sparks(self, tet):
        # Create small outward sparks around the locked tiles
        import random, pygame
        particles = []
        for j,row in enumerate(tet.shape):
            for i,val in enumerate(row):
                if not val: continue
                x, y = tet.x+i, tet.y+j
                if y < 0 or x < 0 or x >= GRID_W or y >= GRID_H: continue
                cx = x*CELL + CELL//2
                cy = y*CELL + CELL//2
                for _ in range(2):
                    vx = random.uniform(-1.2,1.2)
                    vy = random.uniform(-2.0,-0.2)
                    particles.append({"x":cx, "y":cy, "vx":vx, "vy":vy, "r":2, "life_ms": SPARK_MS, "born_ms":0})
        if particles:
            self.effects.append(Effect('land_sparks', pygame.time.get_ticks(), particles=particles, duration_ms=SPARK_MS))

    def _spawn_rotate_sparks(self):
        import random, pygame
        particles = []
        for j,row in enumerate(self.current.shape):
            for i,val in enumerate(row):
                if not val: continue
                x, y = self.current.x+i, self.current.y+j
                if y < 0 or x < 0 or x >= GRID_W or y >= GRID_H: continue
                base_pts = [
                    (x*CELL, y*CELL), (x*CELL + CELL, y*CELL),
                    (x*CELL, y*CELL + CELL), (x*CELL + CELL, y*CELL + CELL)
                ]
                for (px,py) in base_pts:
                    vx = random.uniform(-0.6,0.6)
                    vy = random.uniform(-0.6,0.6)
                    particles.append({"x":px, "y":py, "vx":vx, "vy":vy, "r":1, "life_ms": SPARK_MS, "born_ms":0})
        if particles:
            self.effects.append(Effect('rotate_sparks', pygame.time.get_ticks(), particles=particles, duration_ms=SPARK_MS))


    def _valid(self, t: Tetromino) -> bool:
        for j,row in enumerate(t.shape):
            for i,val in enumerate(row):
                if not val: continue
                x, y = t.x+i, t.y+j
                if x<0 or x>=GRID_W or y>=GRID_H: return False
                if y>=0 and self.grid[y][x] is not None: return False
        return True

    def _lock(self):
        for j,row in enumerate(self.current.shape):
            for i,val in enumerate(row):
                if not val: continue
                x, y = self.current.x+i, self.current.y+j
                if y<0: self.game_over=True; return
                self.grid[y][x] = PIECE_COLORS[ORDER.index(self.current.name)]
        self._after_lock()

    
    def _after_lock(self):
        # Detect full rows
        full = [y for y in range(GRID_H) if all(self.grid[y][x] is not None for x in range(GRID_W))]
        n = len(full)

        # Tiny land sparks at contact points (regardless of clears)
        self._spawn_land_sparks(self.current)

        if n and not self.reduce_motion:
            # Start line clear animation; postpone deletion & scoring
            now = pygame.time.get_ticks()
            self.clearing_rows = full[:]
            self.clearing_started = now
            self.clearing_score_pending = n
            self.effects.append(Effect('line_clear', now, rows=full, duration_ms=LINE_CLEAR_MS))
            # Screen shake if Tetris
            if n == 4:
                self.shake_until = now + SHAKE_MS
                self.shake_mag = SHAKE_MAG
            # Do NOT spawn the next piece yet; wait for update() to finish clear
            return
        else:
            # Either no clears or reduce_motion is on: do instant clear and scoring
            if n:
                for y in reversed(full):
                    del self.grid[y]
                    self.grid.insert(0, [None for _ in range(GRID_W)])
                self.lines_total += n
                self.score += LINE_SCORES.get(n, 100*n)
                new_level = self.lines_total // 10
                self._on_level_change(new_level)
                if n >= 2:
                    effect = random.choice(['smiley','confetti','flash','sunglasses','sparkle'])
                    self.effects.append(Effect(effect, pygame.time.get_ticks()))
                if n == 4 and not self.reduce_motion:
                    now = pygame.time.get_ticks()
                    self.shake_until = now + SHAKE_MS
                    self.shake_mag = SHAKE_MAG

        # Spawn the next piece
        self.current = Tetromino(self.next_name, 0, (GRID_W - len(ROTS[self.next_name][0][0]))//2, -2)
        self.next_name = self.bag.next()
        if not self._valid(self.current): self.game_over = True


    
    def _rotate_current(self, d=1):
        old = Tetromino(self.current.name, self.current.ri, self.current.x, self.current.y)
        self.current.rotate(d)
        kicks = [(0,0),(-1,0),(1,0),(0,-1)]
        rotated_ok = False
        for dx,dy in kicks:
            self.current.x += dx; self.current.y += dy
            if self._valid(self.current):
                rotated_ok = True
                break
            self.current.x -= dx; self.current.y -= dy
        if not rotated_ok:
            self.current = old
        else:
            # rotation success: emit subtle sparks
            self._spawn_rotate_sparks()

    def _move(self, dx:int):
        self.current.x += dx
        if not self._valid(self.current): self.current.x -= dx

    def _soft_drop_tick(self):
        now = pygame.time.get_ticks()
        interval = SOFT_DROP_MS if self.soft_drop else self.drop_ms
        if now - self.last_drop >= interval:
            self.current.y += 1
            if not self._valid(self.current):
                self.current.y -= 1
                self._lock()
            self.last_drop = now

    # ---- layout ----
    def _layout_sidebar(self):
        x0 = LEFTBAR_W + GRID_W*CELL + MARGIN*2
        bw = SIDEBAR_W - 2*MARGIN
        bh, gap = 42, 14
        # Compute y just below the next piece preview
        y = MARGIN + 60          # title
        y += 24*3                # Score/Lines/Level
        y += 10 + 8              # spacing + "Next:" label
        y += 120                 # next-preview box height
        y += 12
        for idx, name in enumerate(['Start','Pause','Reset','Exit']):
            b = self.buttons[name]
            b.rect.x = x0; b.rect.y = y + idx*(bh+gap)
            b.rect.w = bw; b.rect.h = bh

    # ---- input ----
    def handle_mouse(self, pos, pressed):
        # Handle overlay save/skip clicks first
        if self.game_over and self.name_entry_active and pressed[0]:
            if self.overlay_save_rect and self.overlay_save_rect.collidepoint(pos):

                if not self.highscore_submitted:

                    self.scores.add(self.name_buffer or "Player", self.score)

                    self.highscore_submitted = True

                    self.name_entry_active = False
                return
            if self.overlay_skip_rect and self.overlay_skip_rect.collidepoint(pos):
                self.name_entry_active = False
                return

        self._layout_sidebar()
        for name,b in self.buttons.items():
            b.hover = b.rect.collidepoint(pos)
            if b.hover and pressed[0]:
                if name=='Start': self.start()
                elif name=='Pause': self.pause_toggle()
                elif name=='Reset': self.reset()
                elif name=='Exit': pygame.event.post(pygame.event.Event(pygame.QUIT))

    def handle_key(self, event):
        if event.type == pygame.KEYDOWN:
            if self.game_over:
                if self.name_entry_active:
                    if event.key == pygame.K_RETURN:

                        if not self.highscore_submitted:

                            self.scores.add(self.name_buffer or "Player", self.score)

                            self.highscore_submitted = True

                            self.name_entry_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.name_buffer = self.name_buffer[:-1]
                    else:
                        ch = event.unicode
                        if ch and len(ch)==1 and len(self.name_buffer)<16 and ch.isprintable() and not ch.isspace():
                            self.name_buffer += ch
                else:
                    if event.key in (pygame.K_y, pygame.K_RETURN, pygame.K_SPACE):
                        self.reset(); self.start()
                    elif event.key in (pygame.K_n, pygame.K_ESCAPE):
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
                return

            if not self.started or self.paused:
                if event.key == pygame.K_p: self.pause_toggle()
                if event.key == pygame.K_SPACE and not self.started: self.start()
                return

            if event.key == pygame.K_LEFT:
                self.key_left = True; self.move_dir = -1
                self._move(-1); self.move_repeat_t = pygame.time.get_ticks() + DAS_MS
            elif event.key == pygame.K_RIGHT:
                self.key_right = True; self.move_dir = 1
                self._move(1); self.move_repeat_t = pygame.time.get_ticks() + DAS_MS
            elif event.key == pygame.K_DOWN:
                self.soft_drop = True
            elif event.key in (pygame.K_SPACE, pygame.K_UP):
                self._rotate_current(1)
            elif event.key == pygame.K_p:
                self.pause_toggle()
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN: self.soft_drop = False
            elif event.key == pygame.K_LEFT:
                self.key_left = False
                if self.key_right:
                    self.move_dir = 1; self.move_repeat_t = pygame.time.get_ticks() + DAS_MS
                else:
                    self.move_dir = 0
            elif event.key == pygame.K_RIGHT:
                self.key_right = False
                if self.key_left:
                    self.move_dir = -1; self.move_repeat_t = pygame.time.get_ticks() + DAS_MS
                else:
                    self.move_dir = 0

    # ---- public ----
    def reset(self):
        self.grid = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.bag = Bag7()
        self.current = self._spawn()
        self.next_name = self.bag.next()
        self.level = 0; self.lines_total = 0; self.score = 0
        self.drop_ms = BASE_DROP_MS; self.last_drop = pygame.time.get_ticks()
        self.soft_drop = False; self.effects.clear()
        self.started = False; self.paused = False; self.game_over = False
        self.name_entry_active = False; self.name_buffer = ""
        self.key_left = self.key_right = False; self.move_dir = 0; self.move_repeat_t = 0
        self.bg_index = 0; self._bg_cache_key = None

    def start(self):
        if not self.started: self.started = True; self.paused = False

    def pause_toggle(self):
        if self.started and not self.game_over: self.paused = not self.paused

    def update(self):
        if self.started and not self.paused and not self.game_over:
            self._soft_drop_tick()
            if self.move_dir != 0:
                now = pygame.time.get_ticks()
                if now >= self.move_repeat_t:
                    self._move(self.move_dir)
                    self.move_repeat_t = now + ARR_MS

        if self.game_over and not self.name_entry_active and not self.highscore_submitted:
            if self.scores.qualifies(self.score): self.name_entry_active = True
        # Finish pending line clears after animation
        if self.clearing_rows:
            now = pygame.time.get_ticks()
            if now - self.clearing_started >= LINE_CLEAR_MS or self.reduce_motion:
                # Delete rows, score, level up, effects
                for y in reversed(self.clearing_rows):
                    del self.grid[y]
                    self.grid.insert(0, [None for _ in range(GRID_W)])
                n = len(self.clearing_rows)
                self.lines_total += n
                self.score += LINE_SCORES.get(n, 100*n)
                new_level = self.lines_total // 10
                self._on_level_change(new_level)
                if n >= 2:
                    effect = random.choice(['smiley','confetti','flash','sunglasses','sparkle'])
                    self.effects.append(Effect(effect, pygame.time.get_ticks()))
                # Spawn next piece after clear
                self.clearing_rows = []
                self.clearing_score_pending = 0
                self.current = Tetromino(self.next_name, 0, (GRID_W - len(ROTS[self.next_name][0][0]))//2, -2)
                self.next_name = self.bag.next()
                if not self._valid(self.current): self.game_over = True


    # ---- drawing ----
    def _draw_grid(self, surf):
        pygame.draw.rect(surf, DARK, (0,0,GRID_W*CELL, GRID_H*CELL), border_radius=10)

        # settled tiles
        for y in range(GRID_H):
            for x in range(GRID_W):
                c = self.grid[y][x]
                if c is not None:
                    r = pygame.Rect(x*CELL, y*CELL, CELL-1, CELL-1)
                    pygame.draw.rect(surf, c, r, border_radius=4)

        # subtle grid lines
        for x in range(GRID_W+1):
            pygame.draw.line(surf, (40,40,50), (x*CELL,0), (x*CELL,GRID_H*CELL))
        for y in range(GRID_H+1):
            pygame.draw.line(surf, (40,40,50), (0,y*CELL), (GRID_W*CELL,y*CELL))

        # ghost
        ghost = Tetromino(self.current.name, self.current.ri, self.current.x, self.current.y)
        while self._valid(ghost): ghost.y += 1
        ghost.y -= 1
        for j,row in enumerate(ghost.shape):
            for i,val in enumerate(row):
                if not val: continue
                gx, gy = ghost.x+i, ghost.y+j
                if gy >= 0:
                    r = pygame.Rect(gx*CELL, gy*CELL, CELL-1, CELL-1)
                    pygame.draw.rect(surf, (100,100,120), r, width=2, border_radius=4)

        # current piece
        col = PIECE_COLORS[ORDER.index(self.current.name)]
        for j,row in enumerate(self.current.shape):
            for i,val in enumerate(row):
                if not val: continue
                x, y = self.current.x+i, self.current.y+j
                if y >= 0:
                    r = pygame.Rect(x*CELL, y*CELL, CELL-1, CELL-1)
                    pygame.draw.rect(surf, col, r, border_radius=4)

    def _draw_preview(self, surf, name, box):
        shape = ROTS[name][0]
        w, h = len(shape[0]), len(shape)
        cell = 20
        px = box.x + (box.w - w*cell)//2
        py = box.y + (box.h - h*cell)//2
        col = PIECE_COLORS[ORDER.index(name)]
        for j,row in enumerate(shape):
            for i,val in enumerate(row):
                if not val: continue
                r = pygame.Rect(px + i*cell, py + j*cell, cell-1, cell-1)
                pygame.draw.rect(surf, col, r, border_radius=4)

    def _draw_leftpanel(self, surf):
        # Background
        pygame.draw.rect(surf, (34,34,44), (MARGIN, MARGIN, LEFTBAR_W, GRID_H*CELL - 2*MARGIN), border_radius=16)
        x0 = MARGIN*2
        y = MARGIN + 12

        # Keys
        surf.blit(self.fonts['title_sm'].render("INFO", True, LIGHT), (x0, y)); y += 34
        surf.blit(self.fonts['small'].render("Keys:", True, LIGHT), (x0, y)); y += 22
        for s in ["←/→ move", "↓  soft drop", "SPACE rotate", "P pause", "R reset", "Esc exit"]:
            surf.blit(self.fonts['tiny'].render(s, True, DIM), (x0, y)); y += 18

        # Highscores
        y += 12
        surf.blit(self.fonts['small'].render("Top 10 Scores:", True, LIGHT), (x0, y)); y += 22
        if not self.scores.scores:
            surf.blit(self.fonts['tiny'].render("No scores yet — be the first!", True, DIM), (x0, y))
        else:
            for i, e in enumerate(self.scores.scores[:10]):
                line = f"{i+1:>2}. {e['name'][:12]:<12}  {e['score']}"
                surf.blit(self.fonts['tiny'].render(line, True, DIM), (x0, y)); y += 18

    def _draw_sidebar(self, surf):
        # Right sidebar background
        right_x = LEFTBAR_W + GRID_W*CELL + MARGIN
        pygame.draw.rect(surf, (34,34,44), (right_x, MARGIN, SIDEBAR_W, GRID_H*CELL - 2*MARGIN), border_radius=16)

        x0 = right_x + MARGIN
        title = self.fonts['title'].render("TETRIS", True, LIGHT)
        surf.blit(title, (x0, MARGIN+4))

        y = MARGIN + 60
        for label, value in [("Score", self.score), ("Lines", self.lines_total), ("Level", self.level)]:
            surf.blit(self.fonts['small'].render(f"{label}: {value}", True, LIGHT), (x0, y)); y += 24

        y += 10
        surf.blit(self.fonts['small'].render("Next:", True, DIM), (x0, y)); y += 8
        box = pygame.Rect(x0, y, SIDEBAR_W - 2*MARGIN, 120)
        pygame.draw.rect(surf, (50,50,62), box, border_radius=12)
        self._draw_preview(surf, self.next_name, box)

        # Buttons are drawn using their rects; ensure layout is updated
        for b in self.buttons.values():
            b.draw(surf, self.fonts['btn'])

    def _draw_overlays(self, surf):
        now = pygame.time.get_ticks()
        self.effects = [e for e in self.effects if e.alive(now)]
        for e in self.effects: e.draw(surf, now, self.fonts)

        if self.paused and self.started and not self.game_over:
            overlay = pygame.Surface((GRID_W*CELL, GRID_H*CELL), pygame.SRCALPHA)
            overlay.fill((0,0,0,120)); surf.blit(overlay,(LEFTBAR_W + MARGIN, MARGIN))
            txt = self.fonts['title'].render("PAUSED", True, LIGHT)
            surf.blit(txt, txt.get_rect(center=(LEFTBAR_W + MARGIN + GRID_W*CELL//2, MARGIN + GRID_H*CELL//2)))

        if self.game_over:
            overlay = pygame.Surface((GRID_W*CELL, GRID_H*CELL), pygame.SRCALPHA)
            overlay.fill((0,0,0,160)); surf.blit(overlay,(LEFTBAR_W + MARGIN, MARGIN))
            cx = LEFTBAR_W + MARGIN + GRID_W*CELL//2
            cy = MARGIN + GRID_H*CELL//2
            txt = self.fonts['title'].render("GAME OVER", True, LIGHT)
            surf.blit(txt, txt.get_rect(center=(cx, cy-60)))

            if self.name_entry_active:
                sub = self.fonts['small'].render("New highscore! Enter your name:", True, LIGHT)
                surf.blit(sub, sub.get_rect(center=(cx, cy-12)))
                name = self.fonts['big'].render(self.name_buffer or "_", True, ACCENT)
                surf.blit(name, name.get_rect(center=(cx, cy+20)))

                # Draw Save / Skip buttons
                bw, bh = 120, 44
                gap = 16
                r_save = pygame.Rect(0,0,bw,bh); r_skip = pygame.Rect(0,0,bw,bh)
                r_save.center = (cx - bw//2 - gap//2, cy + 80)
                r_skip.center = (cx + bw//2 + gap//2, cy + 80)
                pygame.draw.rect(surf, (80,120,90), r_save, border_radius=10)
                pygame.draw.rect(surf, (110,80,80), r_skip, border_radius=10)
                surf.blit(self.fonts['btn'].render("Save", True, WHITE), self.fonts['btn'].render("Save", True, WHITE).get_rect(center=r_save.center))
                surf.blit(self.fonts['btn'].render("Skip", True, WHITE), self.fonts['btn'].render("Skip", True, WHITE).get_rect(center=r_skip.center))

                self.overlay_save_rect = r_save
                self.overlay_skip_rect = r_skip
            else:
                sub = self.fonts['small'].render("Play again? (Y/N)", True, LIGHT)
                surf.blit(sub, sub.get_rect(center=(cx, cy)))

    def draw(self):
        self.surface.fill(BLACK)

        # Left info panel
        self._draw_leftpanel(self.surface)

        # Playfield
        play = pygame.Surface((GRID_W*CELL, GRID_H*CELL)).convert_alpha()
        self._draw_grid(play)
        # Screen shake for Tetrises (playfield only)
        bx, by = LEFTBAR_W + MARGIN, MARGIN
        if not self.reduce_motion and pygame.time.get_ticks() < self.shake_until:
            t = (self.shake_until - pygame.time.get_ticks()) / SHAKE_MS
            amp = int(self.shake_mag * max(0, min(1, t)))
            import random
            bx += random.randint(-amp, amp)
            by += random.randint(-amp, amp)
        self.surface.blit(play, (bx, by))

        # ensure we recompute the layout before drawing sidebar
        self._layout_sidebar()
        self._draw_sidebar(self.surface)
        self._draw_overlays(self.surface)


# ------------------------------- Loop ---------------------------------
        # Background overlay at 25% alpha (draw last)
        self._draw_background()

def run_standalone():
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    pygame.display.set_caption("Tetris — Easter Egg")
    W = LEFTBAR_W + GRID_W*CELL + SIDEBAR_W + MARGIN*4
    H = GRID_H*CELL + MARGIN*2

    flags = pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF
    try:
        screen = pygame.display.set_mode((W, H), flags)
    except Exception:
        # Fallback to a small window
        try: screen = pygame.display.set_mode((800, 500))
        except Exception as e2:
            try:
                import ctypes; ctypes.windll.user32.MessageBoxW(0, f"Failed to create game window:\\n{e2}", "Tetris Error", 0)
            except Exception: pass
            raise

    clock = pygame.time.Clock()
    fonts = {
        'title': pygame.font.SysFont(FONT_NAME, 36, bold=True),
        'title_sm': pygame.font.SysFont(FONT_NAME, 28, bold=True),
        'big'  : pygame.font.SysFont(FONT_NAME, 32, bold=True),
        'btn'  : pygame.font.SysFont(FONT_NAME, 22, bold=True),
        'small': pygame.font.SysFont(FONT_NAME, 20),
        'tiny' : pygame.font.SysFont(FONT_NAME, 16),
    }

    scores = Highscores()
    game = TetrisGame(screen, fonts, scores)

    running = True
    while running:
        dt = clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP): game.handle_key(event)

        game.handle_mouse(mouse_pos, mouse_pressed)
        game.update()
        game.draw()
        pygame.display.flip()

    pygame.quit()


def launch_tetris_async():
    """Launch in a separate Python process (non-blocking)."""
    import subprocess
    try:
        module_path = Path(__file__).resolve()
        return subprocess.Popen([sys.executable, str(module_path)], close_fds=True)
    except Exception as e:
        print("Failed to launch Tetris:", e)
        return None


if __name__ == "__main__":
    run_standalone()
