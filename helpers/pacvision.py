#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PacVision — a Pac‑Man style mini‑game (Pygame)
Path: root/helpers/pacvision.py

Backgrounds (overlay @ 15%): root/presets/startup/  — random on start and each new level.
High scores (top 10 only):   root/presets/setsave/pacvision/scores.json
Window: resizable; double‑click the top bar area (within 40 px from top) to maximize/restore (windowed, not fullscreen).
ESC while playing toggles Pause.

Controls: Arrow keys (move), Enter (start), P (pause), M (mute), ESC (pause/unpause during play).
"""

import os, sys, time, random, math, json
from pathlib import Path
import pygame

# --------------- Paths ---------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
BG_DIR = ROOT / "presets" / "startup"
SCORE_DIR = ROOT / "presets" / "setsave" / "pacvision"
SCORE_FILE = SCORE_DIR / "scores.json"

# --------------- Window & Theme ---------------
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
WIDTH, HEIGHT = 960, 1050   # tall so maze fits + header zone
FPS = 60
BG_ALPHA = int(255 * 0.15)  # 15% overlay

# --------------- Game constants ---------------
TILE = 24         # base tile size (scaled per window)
TOP_PAD = 60      # header space for HUD + double-click area

BASE_PAC_SPEED = 8.0        # tiles per second
BASE_GHOST_SPEED = 7.0
POWER_TIME = 12  # legacy; not used directly anymore
BLINK_WINDOW = 5.0  # seconds — ghosts blink before returning to normal

LIVES_START = 3

# --------------- Maze layout ---------------
# Legend:
# # wall, . pellet, o power, ' ' empty, P pac start, G ghost spawn
MAZE_TEXT = [
"############################",
"#............##............#",
"#.####.#####.##.#####.####.#",
"#o####.#####.##.#####.####o#",
"#.####.#####.##.#####.####.#",
"#..........................#",
"#.####.##.########.##.####.#",
"#.####.##.########.##.####.#",
"#......##....##....##......#",
"######.##### ## #####.######",
"     #.##### ## #####.#     ",
"     #.##          ##.#     ",
"     #.## ###--### ##.#     ",
"######.## #      # ##.######",
"      .   #  PG  #   .      ",
"######.## #      # ##.######",
"     #.## ######## ##.#     ",
"     #.##          ##.#     ",
"     #.## ######## ##.#     ",
"######.## ######## ##.######",
"#............##............#",
"#.####.#####.##.#####.####.#",
"#o..##................##..o#",
"###.##.##.########.##.##.###",
"###.##.##.########.##.##.###",
"#......##....##....##......#",
"#.##########.##.##########.#",
"#..........................#",
"############################",
"############################",
]
MAZE_W, MAZE_H = len(MAZE_TEXT[0]), len(MAZE_TEXT)

# --------------- Pygame init ---------------
pygame.init()
pygame.display.set_caption("PacVision")
# Clamp initial window to desktop size (avoid off-screen on high DPI)
try:
    info = pygame.display.Info()
    dw, dh = info.current_w, info.current_h
    WIDTH = min(WIDTH, max(640, dw - 80))
    HEIGHT = min(HEIGHT, max(720, dh - 120))
except Exception:
    pass
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()

FONT = pygame.font.SysFont("consolas", 22)
FONT_BIG = pygame.font.SysFont("consolas", 36, bold=True)
FONT_HUGE = pygame.font.SysFont("consolas", 64, bold=True)

# --------------- Helpers ---------------
def ensure_scores():
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    if not SCORE_FILE.exists():
        SCORE_FILE.write_text("[]", encoding="utf-8")

def load_scores():
    ensure_scores()
    try:
        with open(SCORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def save_score(name, score):
    name = (name or "PLAYER")[:16]
    scores = load_scores()
    # Only allow if in top 10
    if len(scores) >= 10 and score <= min(s["score"] for s in scores):
        return scores
    scores.append({"name": name, "score": int(score), "ts": time.time()})
    scores.sort(key=lambda s: s["score"], reverse=True)
    del scores[10:]
    try:
        with open(SCORE_FILE, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return scores

def draw_text(surf, text, pos, font=FONT, color=(240,240,240), center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center: rect.center = pos
    else: rect.topleft = pos
    surf.blit(img, rect)
    return rect

def list_bgs():
    if not BG_DIR.exists(): return []
    files = []
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    for e in exts:
        files += list(BG_DIR.glob(e))
    return [p for p in files if p.is_file()]

def load_bg_scaled(path):
    try:
        img = pygame.image.load(str(path)).convert_alpha()
        img = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
        img.set_alpha(BG_ALPHA)
        return img
    except Exception:
        return None

def random_bg(exclude=None):
    imgs = list_bgs()
    if not imgs: return None, None
    if exclude in imgs and len(imgs) > 1:
        imgs = [p for p in imgs if p != exclude]
    pick = random.choice(imgs)
    return load_bg_scaled(pick), pick

# Geometry scaling
def compute_tile():
    usable_h = HEIGHT - TOP_PAD
    sz = min(WIDTH // MAZE_W, usable_h // MAZE_H)
    sz = max(14, sz)  # keep playable when small
    offx = (WIDTH - sz*MAZE_W)//2
    offy = TOP_PAD + (usable_h - sz*MAZE_H)//2
    return sz, offx, offy

# --------------- Maze processing ---------------
def parse_maze():
    pellets = set()
    power = set()
    walls = set()
    # Classic-ish default starts (safe distance from ghost pen)
    pac_start = (23, 13)
    ghost_start = (14, 13)
    for r, row in enumerate(MAZE_TEXT):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            elif ch == ".":
                pellets.add((r, c))
            elif ch == "o":
                power.add((r, c))
            # 'P' and 'G' markers in the ascii are ignored (treated as floor)
    return walls, pellets, power, pac_start, ghost_start

WALLS, PELLETS_START, POWER_START, PAC_START, GHOST_START = parse_maze()

# Movement helpers
DIRS = {
    "L": (0,-1),
    "R": (0,1),
    "U": (-1,0),
    "D": (1,0),
}
DIR_ORDER = ["L","R","U","D"]

def add_pos(a,b): return (a[0]+b[0], a[1]+b[1])

def is_wall(rc): return rc in WALLS

def neighbors(rc):
    r,c = rc
    for d,(dr,dc) in DIRS.items():
        t = (r+dr, c+dc)
        if not is_wall(t):
            yield d, t

def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

# --------------- Entities ---------------
class Mover:
    def __init__(self, start_rc, speed_tiles):
        self.rc = start_rc
        self.to_rc = start_rc
        self.progress = 0.0
        self.speed = speed_tiles  # tiles/sec
        self.dir = None  # 'L','R','U','D'

    def at_center(self):
        return self.rc == self.to_rc and self.progress <= 0.0

    def set_dir(self, d):
        dr,dc = DIRS[d]
        nxt = (self.rc[0]+dr, self.rc[1]+dc)
        if not is_wall(nxt):
            self.dir = d
            self.to_rc = nxt
            self.progress = 1.0
            return True
        return False

    def step(self, dt):
        if self.progress > 0.0:
            dist = self.speed * dt
            # convert to tile progress
            self.progress -= dist
            if self.progress <= 0.0:
                self.rc = self.to_rc
                self.progress = 0.0
                return True  # reached tile
        return False

    def pixel_pos(self, tile, offx, offy):
        # linear interpolation between rc and to_rc by (1-progress)
        r0,c0 = self.rc
        r1,c1 = self.to_rc
        t = 1.0 - self.progress
        rr = r0 + (r1-r0)*t
        cc = c0 + (c1-c0)*t
        x = offx + int(cc*tile + tile/2)
        y = offy + int(rr*tile + tile/2)
        return x,y

PAC_SPEED = BASE_PAC_SPEED
GHOST_SPEED = BASE_GHOST_SPEED

class Pac(Mover):
    def __init__(self, rc):
        super().__init__(rc, PAC_SPEED)
        self.want = None  # desired direction from input

class Ghost(Mover):
    def __init__(self, rc, color):
        super().__init__(rc, GHOST_SPEED)
        self.color = color
        self.fright = 0.0  # seconds left

# --------------- Game ---------------

# --------------- Fruit system ---------------
# Fruit types and base weights (rarer fruits are worth more)
FRUIT_TYPES = [
    { "name": "cherry",     "points": 50 },
    { "name": "strawberry", "points": 100 },
    { "name": "orange",     "points": 200 },
    { "name": "apple",      "points": 300 },
    { "name": "key",        "points": 500 },
]
FRUIT_WEIGHTS_BASE = [5, 4, 3, 2, 1]  # 50 most common, 500 rarest
class Game:
    def __init__(self):
        self.state = "START"  # START, RUN, PAUSE, GAMEOVER
        self.time_accum = 0.0
        self.power_fx_timer = 0.0  # radial burst effect after eating a power pellet
        self.offscreen_timer = 0.0  # time Pac-Man has been completely off-screen
        self.level = 1
        self.score = 0
        self.lives = LIVES_START
        self.name_input = ""
        self.scores_snapshot = load_scores()
        self.muted = False

        self.tile, self.offx, self.offy = compute_tile()
        self.bg_surf, self.bg_path = random_bg(exclude=None)

        # Fruit & extra life systems
        self.fruits = []              # list of {'rc': (r,c), 'idx': int, 'points': int, 'ttl': float}
        self.fruit_timer = random.uniform(10.0, 16.0)
        self.block_500_until_level = 0  # 500-pt fruit cooldown for 3 levels after showing
        self.next_extra = 10000         # extra life threshold (stacking)

        # Power-up system
        self.powerup = None          # {'rc': (r,c), 'type': str, 'good': bool, 'ttl': float}
        self.powerup_timer = random.uniform(12.0, 22.0)
        self.powerup_spawned_level = False
        self.pac_shield_timer = 0.0
        self.pac_slow_timer = 0.0

        # Logical canvas (letterboxed)
        self.canvas = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
        self.double_last = 0
        self.toggle_block_until = 0.0
        self.maximized = False
        self.restore_size = (WIDTH, HEIGHT)

        # Initialize timers and level state
        self.ready_timer = 0.0
        self.release_timer = 0.0
        self.release_done = False
        self.reset_level(full=True)


    def rescale_bg(self):
        global WIDTH, HEIGHT, screen
        if self.bg_path:
            try:
                img = pygame.image.load(str(self.bg_path)).convert_alpha()
                img = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
                img.set_alpha(BG_ALPHA)
                self.bg_surf = img
            except Exception:
                self.bg_surf = None

    def new_level_bg(self):
        self.bg_surf, self.bg_path = random_bg(exclude=self.bg_path)

    # ----- Input / Window helpers -----
    def toggle_maximize(self):
        global WIDTH, HEIGHT, screen
        if not getattr(self, 'maximized', False):
            self.restore_size = screen.get_size()
            info = pygame.display.Info()
            dw, dh = info.current_w, info.current_h
            WIDTH = max(800, dw - 80)
            HEIGHT = max(600, dh - 120)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            self.maximized = True
        else:
            WIDTH, HEIGHT = self.restore_size
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            self.maximized = False
        # Refresh surfaces and layout after size change
        self.canvas = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
        self.tile, self.offx, self.offy = compute_tile()
        self.rescale_bg()

    def try_pac_turn(self):
        if self.pac.want:
            self.pac.set_dir(self.pac.want)

    def ghost_next_dir(self, g: Ghost):
        # prefer direction reducing manhattan distance; avoid reversing
        options = []
        r,c = g.rc
        for d,(dr,dc) in DIRS.items():
            # can't reverse unless dead end
            if g.dir and ((d == "L" and g.dir=="R") or (d=="R" and g.dir=="L") or (d=="U" and g.dir=="D") or (d=="D" and g.dir=="U")):
                continue
            t = (r+dr, c+dc)
            if not is_wall(t):
                options.append((d,t))
        if not options:
            return None
        # frightened: random choice
        if g.fright > 0.0:
            return random.choice(options)[0]
        # chase: choose option minimizing distance to pac
        px = self.pac.rc
        best = min(options, key=lambda it: manhattan(it[1], px))
        # add a bit of randomness
        if random.random() < 0.15 and len(options) > 1:
            return random.choice(options)[0]
        return best[0]

    def find_gate_cell(self):
        # Locate the ghost pen gate marked by '-' in MAZE_TEXT (take the leftmost if multiple)
        for r, row in enumerate(MAZE_TEXT):
            for c, ch in enumerate(row):
                if ch == '-':
                    return (r, c)
        # Fallback: just above ghost start
        return (GHOST_START[0]-1, GHOST_START[1])

    def choose_fruit_index(self):
        # Weighted random choice; 500-pt fruit blocked during cooldown
        avail = list(range(len(FRUIT_TYPES)))
        weights = list(FRUIT_WEIGHTS_BASE)
        # Index 4 corresponds to the 500-point fruit in FRUIT_TYPES
        if self.level < self.block_500_until_level and 4 in avail:
            i = avail.index(4)
            avail.pop(i); weights.pop(i)
        if not avail:
            avail, weights = [0,1,2,3], [5,4,3,2]
        total = sum(weights)
        r = random.uniform(0, total)
        c = 0.0
        for idx, w in zip(avail, weights):
            c += w
            if r <= c:
                return idx
        return avail[-1]


    def spawn_fruit(self):
        # Spawn a single fruit (same rules as before) into self.fruits with its own TTL
        for _ in range(300):
            r = random.randrange(MAZE_H); c = random.randrange(MAZE_W)
            if is_wall((r, c)): continue
            if manhattan((r, c), GHOST_START) <= 2: continue
            if hasattr(self, 'pac') and (r, c) == self.pac.rc: continue
            idx = self.choose_fruit_index()
            points = FRUIT_TYPES[idx]['points']
            ttl = 20.0 if points <= 200 else 15.0
            self.fruits.append({'rc': (r, c), 'idx': idx, 'points': points, 'ttl': ttl})
            if points == 500: self.block_500_until_level = self.level + 4
            return True
        return False

    def spawn_n_fruits(self, n):
        spawned = 0
        tried = 0
        while spawned < n and tried < 600:
            if self.spawn_fruit():
                spawned += 1
            tried += 1
        return spawned

    def apply_speed_modifiers(self):
        # Recompute current speeds each frame from base + modifiers
        base_pac = getattr(self, '_pac_base_speed', BASE_PAC_SPEED)
        base_ghost = getattr(self, '_ghost_base_speed', BASE_GHOST_SPEED)
        pac_speed = base_pac
        if self.pac_slow_timer > 0.0:
            pac_speed *= 0.90  # 10% slower
        self.pac.speed = pac_speed
        for g in self.ghosts:
            g.speed = base_ghost

    def spawn_extra_ghost(self):
        # Create a temporary ghost that lasts until Pac is caught or level ends
        g = Ghost(GHOST_START, (200,200,200))
        g.is_temp = True
        g.released_out = False
        g.fright = 0.0
        # If release already occurred, place below gate and move up
        gr, gc = self.gate_rc
        below = (gr+1, gc)
        g.rc = below; g.to_rc = below; g.progress = 0.0; g.dir = None
        g.set_dir('U')
        self.ghosts.append(g)

    def spawn_powerup(self):
        # Decide type: 3 good, 2 bad
        types = ['GOOD_FRUITS', 'GOOD_SHIELD', 'GOOD_SCARE', 'BAD_EXTRA_GHOST', 'BAD_SLOW']
        t = random.choice(types)
        good = t.startswith('GOOD')
        # pick location
        for _ in range(300):
            r = random.randrange(MAZE_H); c = random.randrange(MAZE_W)
            if is_wall((r, c)): continue
            if manhattan((r, c), GHOST_START) <= 2: continue
            if hasattr(self, 'pac') and (r, c) == self.pac.rc: continue
            # avoid overlapping fruits
            if any(fr['rc']==(r,c) for fr in self.fruits): continue
            self.powerup = {'rc': (r, c), 'type': t, 'good': good, 'ttl': 15.0}
            return True
        return False

    def handle_powerup_pick(self):
        if not self.powerup: return
        prc = self.powerup['rc']
        if self.pac.rc != prc: return
        t = self.powerup['type']
        # Apply effects
        if t == 'GOOD_FRUITS':
            self.spawn_n_fruits(3)
        elif t == 'GOOD_SHIELD':
            self.pac_shield_timer = 15.0
        elif t == 'GOOD_SCARE':
            dur = self.get_power_time()
            for gg in self.ghosts: gg.fright = dur
            self.power_fx_timer = 1.25
        elif t == 'BAD_EXTRA_GHOST':
            self.spawn_extra_ghost()
        elif t == 'BAD_SLOW':
            self.pac_slow_timer = 10.0
        # Clear power-up and mark as used this level
        self.powerup = None
        self.powerup_spawned_level = True


    
    def get_power_time(self):
        """Return frightened duration (seconds) for current level.
        Level 1: 30s, then -1s per level until 20s minimum (level 11+)."""
        return max(20.0, 30.0 - (self.level - 1))
    
    def _find_edge_walkable(self, row, side):
        """Return a walkable column on the given row near the requested side.
        side: 'left' -> smallest c; 'right' -> largest c. Fallback to PAC_START col."""
        row = max(0, min(MAZE_H-1, row))
        if side == 'left':
            rng = range(0, MAZE_W)
        else:
            rng = range(MAZE_W-1, -1, -1)
        for c in rng:
            if not is_wall((row, c)):
                return c
        # Fallback: try PAC_START column if that row is blocked entirely
        fallback_c = PAC_START[1] if 0 <= PAC_START[1] < MAZE_W else MAZE_W//2
        return fallback_c
    def check_extra_life(self):
            # Grant extra lives at each 10,000 points milestone (stacking)
            while self.score >= self.next_extra:
                self.lives += 1
                self.next_extra += 10000
    

    def reset_level(self, full=False):
            # Refill pellets & power only on full reset (new level / new game)
            if full or not hasattr(self, 'pellets'):
                self.pellets = set(PELLETS_START)
                self.power = set(POWER_START)

            # Cache gate cell
            if not hasattr(self, 'gate_rc'):
                self.gate_rc = self.find_gate_cell()

            # Reposition entities
            self.pac = Pac(PAC_START)
            colors = [(255,0,0),(255,128,0),(0,255,255),(255,105,180)]
            self.ghosts = [Ghost(GHOST_START, c) for c in colors]
            for g in self.ghosts:
                g.fright = 0.0
                g.released_out = False  # becomes True once above the gate
                g.is_temp = False       # for bad power-up spawned ghosts

            # Speeds — base scale rising by level, with Pac advantage that starts at +20% and decreases 1%/level, min +5%
            speed_scale = 0.40 + 0.03 * (self.level - 1)
            ghost_base = BASE_GHOST_SPEED * speed_scale
            advantage = max(0.05, 0.20 - 0.01 * (self.level - 1))
            pac_base = ghost_base * (1.0 + advantage)
            self._ghost_base_speed = ghost_base
            self._pac_base_speed = pac_base
            # Apply immediately (modifiers may adjust per-frame)
            self.pac.speed = pac_base
            for g in self.ghosts:
                g.speed = ghost_base

            # Start-of-life grace & ghost release after 3 seconds
            self.ready_timer = 1.8
            self.release_timer = 3.0
            self.release_done = False

            # Fruits list and timer
            self.fruits = []
            self.fruit_timer = random.uniform(8.0, 14.0) if not full else random.uniform(10.0, 16.0)

            # Temporary effects & power-ups
            if not full:
                # On life loss: remove temporary extra ghosts and clear shield
                self.ghosts = [g for g in self.ghosts if not getattr(g, 'is_temp', False)]
                self.pac_shield_timer = 0.0
            else:
                # New level: allow one power-up to spawn this level
                self.powerup = None
                self.powerup_spawned_level = False
                self.powerup_timer = random.uniform(12.0, 22.0)

    def draw_maze(self, surf):
            # draw walls as solid tiles
            t, ox, oy = self.tile, self.offx, self.offy
            for r in range(MAZE_H):
                for c in range(MAZE_W):
                    if (r, c) in WALLS:
                        pygame.draw.rect(surf, (30, 60, 120), (ox + c*t, oy + r*t, t, t))
            # pellets
            for (r, c) in self.pellets:
                pygame.draw.circle(surf, (250, 220, 120), (ox + c*t + t//2, oy + r*t + t//2), max(2, t//8))
            # power pellets
            for (r, c) in self.power:
                pygame.draw.circle(surf, (255, 255, 255), (ox + c*t + t//2, oy + r*t + t//2), max(4, t//4), 2)
    
    def draw_fruit(self, surf):
            if not self.fruits:
                return
            t, ox, oy = self.tile, self.offx, self.offy
            colors = [(230,60,60), (255,100,180), (255,170,40), (90,200,90), (250,250,120)]
            for fr in self.fruits:
                r, c = fr['rc']
                x, y = ox + c*t + t//2, oy + r*t + t//2
                idx = fr['idx']
                color = colors[idx % len(colors)]
                pygame.draw.circle(surf, color, (x, y), max(5, t//3))
                pygame.draw.circle(surf, (40,120,40), (x - t//4, y - t//3), max(2, t//10))

    
    
    
    
        
    
        
    def draw_power_fx(self, surf):
        """Draw a brief radial burst around Pac-Man after picking a power pellet."""
        if self.power_fx_timer <= 0.0:
            return
        t, ox, oy = self.tile, self.offx, self.offy
        px, py = self.pac.pixel_pos(t, ox, oy)
        elapsed = 1.25 - self.power_fx_timer
        # Screen flash (subtle)
        flash_alpha = int(90 * (self.power_fx_timer / 1.25))
        if flash_alpha > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 200, flash_alpha))
            surf.blit(overlay, (0, 0))
        # Three expanding rings
        for i in range(3):
            prog = (elapsed - i * 0.12) / 0.60
            if prog <= 0.0 or prog > 1.0:
                continue
            radius = int((t * 1.6) + prog * t * 3.6)
            alpha = int(255 * (1.0 - prog))
            ring = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring, (255, 250, 180, alpha), (radius + 2, radius + 2), radius, 4)
            surf.blit(ring, (px - radius - 2, py - radius - 2))
    
    
    def draw_entities(self, surf):
            t, ox, oy = self.tile, self.offx, self.offy
            # pac
            px,py = self.pac.pixel_pos(t, ox, oy)
            pygame.draw.circle(surf, (255, 228, 88), (px,py), t//2 - 2)
            # mouth
            ang = {"L":(200,320), "R":(20,140), "U":(110,230), "D":(290,410)}.get(self.pac.dir or "L",(30,330))
            start,end = math.radians(ang[0]), math.radians(ang[1])
            pygame.draw.arc(surf, (22,22,22), (px-(t//2-2), py-(t//2-2), t-4, t-4), start, end, 4)

            # Shield aura
            if self.pac_shield_timer > 0.0:
                prog = (math.sin(self.time_accum * 6.0) + 1.0) * 0.5  # pulse
                radius = int(t*0.7 + prog * t*0.2)
                alpha = 120 + int(prog * 80)
                ring = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(ring, (120, 240, 160, alpha), (radius + 2, radius + 2), radius, 3)
                surf.blit(ring, (px - radius - 2, py - radius - 2))

            # ghosts
            for g in self.ghosts:
                gx,gy = g.pixel_pos(t, ox, oy)
                color = g.color
                if g.fright > 0.0:
                    blue = (70, 130, 180)
                    if g.fright <= BLINK_WINDOW:
                        blink = (int(self.time_accum * 6) % 2) == 0
                        color = g.color if blink else blue
                    else:
                        color = blue
                body = pygame.Rect(gx-(t//2-2), gy-(t//2-2), t-4, t-2)
                pygame.draw.rect(surf, color, body, border_radius=8)
                # eyes
                pygame.draw.circle(surf, (255,255,255), (gx-6, gy-4), 4)
                pygame.draw.circle(surf, (255,255,255), (gx+6, gy-4), 4)
                pygame.draw.circle(surf, (30,30,30), (gx-6, gy-4), 2)
                pygame.draw.circle(surf, (30,30,30), (gx+6, gy-4), 2)

            # Power-up marker (flashing)
            if self.powerup:
                r, c = self.powerup['rc']
                x, y = ox + c*t + t//2, oy + r*t + t//2
                good = self.powerup['good']
                # flash between bright and dim
                flash = (int(self.time_accum * 8) % 2) == 0
                col = (60, 220, 80) if good else (230, 70, 70)
                col2 = (180, 255, 190) if good else (255, 160, 160)
                base_col = col2 if flash else col
                pygame.draw.circle(surf, base_col, (x, y), max(6, t//3), 4)
                pygame.draw.circle(surf, base_col, (x, y), max(10, t//2), 2)

    def draw_hud(self, surf):
            draw_text(surf, f"Score: {self.score}", (16, 10))
            draw_text(surf, f"Level: {self.level}", (200, 10))
            draw_text(surf, f"Lives: {self.lives}", (360, 10))
    def update(self, dt):
        if self.state != "RUN":
            return
        # clock
        self.time_accum += dt
        if self.power_fx_timer > 0.0:
            self.power_fx_timer = max(0.0, self.power_fx_timer - dt)

        # Timed modifiers
        if self.pac_shield_timer > 0.0:
            self.pac_shield_timer = max(0.0, self.pac_shield_timer - dt)
        if self.pac_slow_timer > 0.0:
            self.pac_slow_timer = max(0.0, self.pac_slow_timer - dt)

        # Apply speed modifiers continuously
        self.apply_speed_modifiers()

        # Off-screen wrap
        px, py = self.pac.pixel_pos(self.tile, self.offx, self.offy)
        radius = self.tile // 2
        outside = (px < -radius) or (px > WIDTH + radius) or (py < -radius) or (py > HEIGHT + radius)
        if outside:
            self.offscreen_timer += dt
            if self.offscreen_timer >= 1.6:
                side = 'left' if px < 0 else 'right'
                row = self.pac.rc[0]
                target_c = self._find_edge_walkable(row, 'right' if side == 'left' else 'left')
                self.pac.rc = (row, target_c)
                self.pac.to_rc = (row, target_c)
                self.pac.progress = 0.0
                self.offscreen_timer = 0.0
        else:
            self.offscreen_timer = 0.0

        # Level start timers
        if self.ready_timer > 0.0:
            self.ready_timer = max(0.0, self.ready_timer - dt)
        if self.release_timer > 0.0:
            self.release_timer = max(0.0, self.release_timer - dt)
        if self.release_timer == 0.0 and not getattr(self, 'release_done', False):
            gr, gc = self.gate_rc
            below_gate = (gr+1, gc)
            for g in self.ghosts:
                g.rc = below_gate
                g.to_rc = below_gate
                g.progress = 0.0
                g.dir = None
                g.set_dir('U')  # force upwards through the gate
                g.released_out = False
            self.release_done = True

        # Fruit timers (list)
        if not self.fruits:
            if self.ready_timer <= 0.0 and self.release_timer <= 0.0:
                self.fruit_timer -= dt
                if self.fruit_timer <= 0.0:
                    if self.spawn_fruit():
                        self.fruit_timer = random.uniform(15.0, 25.0)
                    else:
                        self.fruit_timer = random.uniform(6.0, 10.0)
        else:
            for fr in list(self.fruits):
                fr['ttl'] -= dt
                if fr['ttl'] <= 0.0:
                    self.fruits.remove(fr)

        # Power-up timers
        if not self.powerup and not self.powerup_spawned_level and self.ready_timer <= 0.0 and self.release_timer <= 0.0:
            self.powerup_timer -= dt
            if self.powerup_timer <= 0.0:
                if not self.spawn_powerup():
                    self.powerup_timer = random.uniform(8.0, 14.0)
        elif self.powerup:
            self.powerup['ttl'] -= dt
            if self.powerup['ttl'] <= 0.0:
                self.powerup = None
                self.powerup_spawned_level = True

        # Pac input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: self.pac.want = "L"
        elif keys[pygame.K_RIGHT]: self.pac.want = "R"
        elif keys[pygame.K_UP]: self.pac.want = "U"
        elif keys[pygame.K_DOWN]: self.pac.want = "D"

        # try to turn when centered
        if self.pac.at_center():
            self.try_pac_turn()
            if self.pac.dir is None and self.pac.want:
                self.pac.set_dir(self.pac.want)

        # step Pac
        reached = self.pac.step(dt)

        # Ghost movement
        gr, gc = self.gate_rc
        for g in self.ghosts:
            if self.ready_timer <= 0.0 and self.release_timer <= 0.0:
                if g.fright > 0.0:
                    g.fright = max(0.0, g.fright - dt)

                if not g.released_out:
                    if g.at_center():
                        if g.rc[0] >= gr - 1:
                            if not g.set_dir('U'):
                                if g.rc[1] < gc: g.set_dir('R')
                                elif g.rc[1] > gc: g.set_dir('L')
                                else:
                                    for d,_ in neighbors(g.rc):
                                        if d != 'D':
                                            if g.set_dir(d): break
                        else:
                            g.released_out = True
                    g.step(dt)
                else:
                    if g.at_center():
                        nd = self.ghost_next_dir(g)
                        if nd:
                            g.set_dir(nd)
                    g.step(dt)

        # Tile arrival: pellets/power/fruit/powerup
        if reached:
            if self.pac.rc in self.pellets:
                self.pellets.remove(self.pac.rc)
                self.score += 10
                self.check_extra_life()
            if self.pac.rc in self.power:
                self.power.remove(self.pac.rc)
                self.score += 50
                dur = self.get_power_time()
                for gg in self.ghosts:
                    gg.fright = dur
                self.power_fx_timer = 1.25
                self.check_extra_life()

            for fr in list(self.fruits):
                if self.pac.rc == fr['rc']:
                    self.score += fr['points']
                    self.fruits.remove(fr)
                    self.fruit_timer = random.uniform(15.0, 25.0)
                    self.check_extra_life()

            self.handle_powerup_pick()

        # collisions (ghosts)
        pac_xy = self.pac.pixel_pos(self.tile, self.offx, self.offy)
        for g in self.ghosts:
            gx, gy = g.pixel_pos(self.tile, self.offx, self.offy)
            if self.ready_timer <= 0.0 and (abs(gx - pac_xy[0]) < self.tile * 0.6) and (abs(gy - pac_xy[1]) < self.tile * 0.6):
                if g.fright > 0.0:
                    self.score += 200
                    self.check_extra_life()
                    g.rc = GHOST_START
                    g.to_rc = GHOST_START
                    g.progress = 0.0
                    g.dir = None
                    g.fright = 0.0
                    g.released_out = False
                    if self.release_done:
                        gr, gc = self.gate_rc
                        bg = (gr+1, gc)
                        g.rc = bg; g.to_rc = bg; g.progress = 0.0; g.dir = None; g.set_dir('U')
                else:
                    if self.pac_shield_timer > 0.0:
                        continue
                    self.lives -= 1
                    if self.lives <= 0:
                        self.state = "GAMEOVER"
                    else:
                        self.reset_level(full=False)
                    return

        # level complete?
        if not self.pellets and not self.power:
            self.level += 1
            self.reset_level(full=True)
            self.new_level_bg()

    def handle(self, event):
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == "RUN":
                        self.state = "PAUSE"
                    elif self.state == "PAUSE":
                        self.state = "RUN"
                if event.key == pygame.K_RETURN:
                    if self.state in ("START", "GAMEOVER"):
                        # start/restart
                        self.level = 1
                        self.score = 0
                        self.lives = LIVES_START
                        self.next_extra = 10000
                        self.state = "RUN"
                        self.reset_level(full=True)
                    elif self.state == "PAUSE":
                        self.state = "RUN"
    
    def draw(self, surf):
            surf.fill((10,10,12))
            # core gameplay
            self.draw_maze(surf)
            self.draw_fruit(surf)
            self.draw_entities(surf)
            # power pellet pickup fx overlay
            self.draw_power_fx(surf)
            self.draw_hud(surf)
    
            # overlay background @15% on TOP of the game
            if self.bg_surf:
                surf.blit(self.bg_surf, (0,0))
    
            if self.state == "PAUSE":
                draw_text(surf, "PAUSED", (WIDTH//2, HEIGHT//2-24), font=FONT_HUGE, center=True)
            if self.state == "RUN" and (self.ready_timer > 0.0 or self.release_timer > 0.0):
                draw_text(surf, "READY!", (WIDTH//2, HEIGHT//2-24), font=FONT_BIG, center=True)
    
            if self.state == "START":
                draw_text(surf, "PACVISION", (WIDTH//2, HEIGHT//2-200), font=FONT_HUGE, center=True)
                lines = [
                    "Eat all pellets, avoid ghosts.",
                    "Power pellets scare ghosts (eat them!).",
                    "Arrows to move, ESC toggles Pause.",
                    "Double‑click top bar to maximize.",
                    "Press ENTER to start",
                ]
                for i, t in enumerate(lines):
                    draw_text(surf, t, (WIDTH//2, HEIGHT//2-60 + i*30), center=True)
                # show highscores
                scores = load_scores()
                draw_text(surf, "Top 10", (WIDTH//2, HEIGHT//2+120), font=FONT_BIG, center=True)
                for i,s in enumerate(scores[:10]):
                    draw_text(surf, f"{i+1:>2}. {s['name'][:12]:<12} {s['score']:>6}", (WIDTH//2-120, HEIGHT//2+150+i*22))
    
            if self.state == "GAMEOVER":
                draw_text(surf, "GAME OVER", (WIDTH//2, HEIGHT//2-140), font=FONT_HUGE, center=True)
                draw_text(surf, f"Score: {self.score}", (WIDTH//2, HEIGHT//2-90), font=FONT_BIG, center=True)
                draw_text(surf, "Enter your name:", (WIDTH//2, HEIGHT//2-40), font=FONT_BIG, center=True)
                nm = self.name_input or "PLAYER"
                pygame.draw.rect(surf, (240,240,240), (WIDTH//2-160, HEIGHT//2-8, 320, 44), 2, border_radius=10)
                draw_text(surf, nm, (WIDTH//2, HEIGHT//2+2), center=True)
                draw_text(surf, "Press ENTER to save, R to restart", (WIDTH//2, HEIGHT//2+60), center=True)
                # highscores
                scores = load_scores()
                draw_text(surf, "Top 10", (WIDTH//2, HEIGHT//2+110), font=FONT_BIG, center=True)
                for i,s in enumerate(scores[:10]):
                    draw_text(surf, f"{i+1:>2}. {s['name'][:12]:<12} {s['score']:>6}", (WIDTH//2-120, HEIGHT//2+140+i*22))
    
        # ----- Events -----
    def handle(self, e):
            global WIDTH, HEIGHT, screen
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if e.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = max(640, e.w), max(720, e.h)
                # No set_mode here; Pygame already resized the window
                self.canvas = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
                self.tile, self.offx, self.offy = compute_tile()
                self.rescale_bg()
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                # double click at the top area (first 40 px)
                if e.pos[1] < 40:
                    now = time.time()
                    if now < getattr(self, 'toggle_block_until', 0):
                        pass
                    elif (now - self.double_last) < 0.35:
                        self.toggle_maximize()
                        self.toggle_block_until = now + 0.5
                        self.double_last = 0
                    else:
                        self.double_last = now
            if e.type == pygame.KEYDOWN:
                if self.state == "RUN":
                    if e.key == pygame.K_ESCAPE:
                        self.state = "PAUSE" if self.state == "RUN" else "RUN"
                    elif e.key == pygame.K_p:
                        self.state = "PAUSE"
                elif self.state == "PAUSE":
                    if e.key == pygame.K_ESCAPE or e.key == pygame.K_p:
                        self.state = "RUN"
                elif self.state == "START":
                    if e.key == pygame.K_RETURN:
                        self.state = "RUN"
                elif self.state == "GAMEOVER":
                    if e.key == pygame.K_RETURN:
                        # save only if in top 10
                        save_score(self.name_input.strip() or "PLAYER", self.score)
                        # reset to start screen
                        self.level = 1
                        self.score = 0
                        self.lives = LIVES_START
                        self.name_input = ""
                        self.ready_timer = 0.0
                        self.reset_level(full=True)
                        self.state = "START"
                    elif e.key == pygame.K_r:
                        # quick restart to start screen
                        self.level = 1
                        self.score = 0
                        self.lives = LIVES_START
                        self.name_input = ""
                        self.ready_timer = 0.0
                        self.reset_level(full=True)
                        self.state = "START"
                    elif e.key == pygame.K_BACKSPACE:
                        self.name_input = self.name_input[:-1]
                    else:
                        ch = e.unicode
                        if ch and (ch.isalnum() or ch in (" ","_","-")) and len(self.name_input) < 16:
                            self.name_input += ch
    # --------------- Main ---------------
def main():
    game = Game()
    dt = 0.0
    while True:
        for e in pygame.event.get():
            game.handle(e)

        if game.state == "RUN":
            game.update(dt)

        # Draw to logical canvas, then scale to current window size
        game.draw(game.canvas)
        scaled = pygame.transform.smoothscale(game.canvas, screen.get_size())
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
        dt = clock.tick(FPS)/1000.0

if __name__ == "__main__":
    ensure_scores()
    main()