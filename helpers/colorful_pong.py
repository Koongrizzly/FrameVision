#!/usr/bin/env python3
# Colorful Pong — Last-Hit Power-Ups + friendly tokens (v1.3.3)
# Buttons always visible; help/high-scores overlays pause/resume with ESC;
# high scores saved to ./presets/setsave/; F11 & double-click top bar = maximize;
# level label lowered below pause button.
#
# v1.3.3 changes:
# - Removed paddle trails (clean paddle rendering).
# - Only balls can collect points and power-ups; paddles no longer pick them up.
#
# v1.3.2 changes:
# - Add left/right movement for player, clamped to midline (A/D or Left/Right).
# - Fix indentation so AI block doesn't throw IndentationError.
# - Help overlay updated for new controls.
#
# v1.3.0 changes:
# - Ball slightly smaller
# - Game over if player trails AI by >10
# - Level-up now awards +200 points
# - Added separate always-respawning point tokens (1, 5, 10) independent of normal power-ups
# - Multiball now adds 2 extra balls (3 total)

import warnings
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)
import pygame, math, random, json, os, sys

WIDTH, HEIGHT = 960, 540
FPS = 120
TOPBAR_H = 36
PADDLE_W, PADDLE_H = 16, 110
BALL_SIZE = 12  # was 14 — a little smaller
START_SPEED = 290.0
# Difficulty tuning
MAX_BALL_SPEED = 520.0  # hard cap on ball speed
MAX_BALL_SPEED_PER_LEVEL = 10.0  # small increase per level
HIT_SPEED_MULT = 1.012  # was 1.03
HIT_VY_BASE = 220.0  # was 280
HIT_VY_PER_LEVEL = 8.0  # was 14
MIN_BALL_SPEED = 100.0  # prevents stalls (esp. after magnet ends)
MIN_BALL_SPEED_PER_LEVEL = 5.0
MIN_HORIZONTAL_RATIO = 0.22  # at least 22% of total speed goes into horizontal movement
LEVEL_UP_EVERY = 5  # every 5 player goals -> level up
MAX_LEVEL = 20
AI_BASE_SPEED = 250.0
AI_REACTION = 0.12

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "presets", "setsave")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "pong_highscores.json")

STARTUP_BG_DIR = os.path.join(ROOT_DIR, "presets", "startup")
POWERUP_SIZE = 28
POWERUP_PICKUP_PAD = 20
POWERUP_ATTRACT_RADIUS = 120
POWERUP_ATTRACT_SPEED = 120.0
POWERUP_COOLDOWN_MIN = 6.0
POWERUP_COOLDOWN_MAX = 10.0
POWERUP_TYPES = ("GROW","SHRINK_OPP","SLOW_BALL","MULTI","CURVE","SHIELD","HARD_SMACK","TRIPLE_SIZE","SMALL_BALL")
POWERUP_EXPLAINS = {
    "GROW":"Your paddle grows for a while.",
    "SHRINK_OPP":"Opponent's paddle shrinks for a while.",
    "SLOW_BALL":"Balls move slower for a while.",
    "MULTI":"Adds two more balls!",
    "CURVE":"Curved hits add spin!",
    "SHIELD":"A wall protects your goal for a while.",
    "HARD_SMACK":"For 10s, your hits are stronger for a more powerful rebound.",
    "TRIPLE_SIZE":"Your paddle becomes 3x size for 10s.",
    "SMALL_BALL":"All balls are half size for 10s.",
}

def hsv_to_rgb(h, s=1, v=1):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r*255), int(g*255), int(b*255))
def rainbow(t): return hsv_to_rgb((t*0.12) % 1.0, 0.85, 1.0)
def soft_glow_surf(radius, color, alpha=160):
    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        a = int(alpha * (r/float(radius))**2)
        pygame.draw.circle(s, (*color, a), (radius, radius), r)
    return s


def scale_cover(image, size):
    # Scale 'image' to completely cover 'size' (like CSS background-size: cover),
    # then center-crop the excess.
    tw, th = size
    iw, ih = image.get_size()
    if iw == 0 or ih == 0:
        return pygame.Surface((tw, th)).convert()
    scale = max(tw / iw, th / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    scaled = pygame.transform.smoothscale(image, (nw, nh))
    x = (nw - tw) // 2
    y = (nh - th) // 2
    dest = pygame.Surface((tw, th)).convert()
    dest.blit(scaled, (-x, -y))
    return dest

class Particle:
    __slots__ = ("x","y","vx","vy","life","max_life","color","size","spin","angle")
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        ang = random.uniform(0, 6.283)
        spd = random.uniform(120, 360)
        self.vx = math.cos(ang)*spd; self.vy = math.sin(ang)*spd
        self.life = 0.0; self.max_life = random.uniform(0.25, 0.6)
        self.color = color; self.size = random.randint(2, 5)
        self.spin = random.uniform(-8, 8); self.angle = random.uniform(0, 6.283)
    def update(self, dt):
        self.life += dt; self.x += self.vx * dt; self.y += self.vy * dt
        self.vy += 500 * dt; self.angle += self.spin * dt
        return self.life < self.max_life
    def draw(self, surf):
        t = self.life / self.max_life; a = int(255 * (1.0 - t))
        if a <= 0: return
        pygame.draw.rect(surf, (*self.color, a), (int(self.x), int(self.y), self.size, self.size))

class Trail:
    def __init__(self, length=16): self.history = []; self.length = length
    def add(self, rect, color):
        self.history.append((rect.copy(), color)); 
        if len(self.history) > self.length: self.history.pop(0)
    def draw(self, surf):
        n = len(self.history)
        for i, (r, color) in enumerate(self.history):
            alpha = int(180 * (i+1)/max(1, n))
            pygame.draw.rect(surf, (*color, alpha), r)

def load_scores():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {"best_score": 0, "best_level": 1, "total_played": 0}
def save_scores(data):
    try:
        with open(DATA_FILE, "w") as f: json.dump(data, f, indent=2)
    except Exception: pass

class TextFX:
    def __init__(self, font): self.font = font; self.messages = []
    def add(self, text, pos, color, ttl=1.0, rise=-40): self.messages.append([text, list(pos), color, 0.0, ttl, rise])
    def update(self, dt): self.messages = [m for m in self.messages if self._upd(m, dt)]
    def _upd(self, m, dt): m[3]+=dt; m[1][1]+= (m[5]*dt); return (m[3]/m[4])<1.0
    def draw(self, surf):
        for text, pos, color, age, ttl, rise in self.messages:
            t = age/ttl; a = int(255*(1-t)); s = self.font.render(text, True, color); s.set_alpha(a); surf.blit(s, s.get_rect(center=pos))

class BigPopupFX:
    def __init__(self, font, small): self.font=font; self.small=small; self.items=[]
    def add(self, text, subtext, pos, color, ttl=1.25): self.items.append([text, subtext, pos, color, 0.0, ttl])
    def update(self, dt): self.items=[it for it in self.items if self._u(it,dt)]
    def _u(self,it,dt): it[4]+=dt; return it[4]<it[5]
    def draw(self, surf):
        for text, subtext, (x,y), color, age, ttl in self.items:
            t=age/ttl; alpha=max(0,min(255,int(255*(1.0-t)))); scale=1.0+0.2*(1.0-t)
            s=self.font.render(text,True,color); s=pygame.transform.rotozoom(s,0,scale); s.set_alpha(alpha); rect=s.get_rect(center=(x,y))
            panel=pygame.Surface((rect.w+26, rect.h+18), pygame.SRCALPHA); panel.fill((0,0,0,140)); surf.blit(panel, panel.get_rect(center=(x,y))); surf.blit(s, rect)
            if subtext:
                sub=self.small.render(subtext,True,(240,240,240)); sub.set_alpha(max(0,min(255,int(220*(1.0-t))))); surf.blit(sub, sub.get_rect(midtop=(x, rect.bottom+6)))

def segment_intersects_rect(p0, p1, rect):
    x0,y0=p0; x1,y1=p1; dx=x1-x0; dy=y1-y0
    p=[-dx,dx,-dy,dy]; q=[x0-rect.left, rect.right-x0, y0-rect.top, rect.bottom-y0]
    u1,u2=0.0,1.0
    for pi,qi in zip(p,q):
        if pi==0:
            if qi<0: return False
        else:
            t=qi/pi
            if pi<0:
                if t>u2: return False
                if t>u1: u1=t
            else:
                if t<u1: return False
                if t<u2: u2=t
    return True

class UIButton:
    def __init__(self, label, rect): self.label=label; self.rect=rect; self.hot=False
    def draw(self, surf, font, hot=False):
        self.hot=hot; bg=(255,255,255, 110 if hot else 60)
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, (255,255,255,160), self.rect, 1, border_radius=8)
        txt=font.render(self.label, True, (245,245,245)); surf.blit(txt, txt.get_rect(center=self.rect.center))

def main():
    pygame.init(); pygame.display.set_caption("Colorful Pong — last-hit power-ups")
    display=pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    canvas=pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def compute_view_rect():
        sw,sh=display.get_size(); scale=min(sw/WIDTH, sh/HEIGHT); vw,vh=int(WIDTH*scale),int(HEIGHT*scale); vx,vy=(sw-vw)//2,(sh-vh)//2
        return pygame.Rect(vx,vy,vw,vh)
    def window_to_canvas(pos, view_rect):
        x,y=pos
        if not view_rect.collidepoint(pos): return None
        return (int((x-view_rect.x)*(WIDTH/view_rect.w)), int((y-view_rect.y)*(HEIGHT/view_rect.h)))

    is_maximized=False
    def toggle_maximize():
        nonlocal is_maximized, display
        is_maximized=not is_maximized
        if is_maximized:
            info=pygame.display.Info(); display=pygame.display.set_mode((info.current_w, info.current_h), pygame.RESIZABLE)
        else:
            display=pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    last_click_time=0; DOUBLE_CLICK_MS=350

    ball_glow=soft_glow_surf(48,(255,255,255),120)
    big=pygame.font.SysFont("arialrounded",64); mid=pygame.font.SysFont("arialrounded",36); small=pygame.font.SysFont("arialrounded",22)
    scores=load_scores()

    # Load semi-transparent background images from presets/startup (logo_1.jpg, logo_2.jpg, ...)
    bg_images = []
    bg_index = 0
    def load_background_overlays():
        imgs = []
        try:
            # Search common locations for startup backgrounds
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "presets", "startup"),
                os.path.join(os.getcwd(), "presets", "startup"),
                STARTUP_BG_DIR,
            ]
            chosen = None
            for cand in candidates:
                cand = os.path.normpath(cand)
                if os.path.isdir(cand):
                    chosen = cand
                    break
            if chosen:
                files = [fn for fn in os.listdir(chosen) if fn.lower().startswith("logo_") and fn.lower().endswith((".jpg", ".jpeg", ".png"))]
                def sort_key(fn):
                    import re as _re
                    m = _re.search(r"(\d+)", fn)
                    return int(m.group(1)) if m else 0
                for fn in sorted(files, key=sort_key):
                    path = os.path.join(chosen, fn)
                    try:
                        base = pygame.image.load(path).convert()
                        cover = scale_cover(base, (WIDTH, HEIGHT))
                        cover.set_alpha(64)  # 25% opacity
                        imgs.append(cover)
                    except Exception:
                        continue
        except Exception:
            pass
        return imgs

    bg_images = load_background_overlays()


    state="MENU"; bg_phase=0.0
    player=pygame.Rect(40, HEIGHT//2-PADDLE_H//2, PADDLE_W, PADDLE_H)
    ai=pygame.Rect(WIDTH-40-PADDLE_W, HEIGHT//2-PADDLE_H//2, PADDLE_W, PADDLE_H)

    balls=[]

    # Difficulty modes
    difficulty_modes = ["Beginner", "Novice", "Daily Player", "Expert"]
    difficulty_index = 0  # default to Beginner
    # Preset tuning per mode
    DIFFICULTY_SETTINGS = {
        "Beginner": dict(START_SPEED=255.0, AI_BASE_SPEED=210.0,
                         MAX_BALL_SPEED=500.0, MAX_BALL_SPEED_PER_LEVEL=8.0,
                         HIT_SPEED_MULT=1.008, HIT_VY_BASE=200.0, HIT_VY_PER_LEVEL=6.0),
        "Novice": dict(START_SPEED=270.0, AI_BASE_SPEED=240.0,
                       MAX_BALL_SPEED=520.0, MAX_BALL_SPEED_PER_LEVEL=10.0,
                       HIT_SPEED_MULT=1.012, HIT_VY_BASE=220.0, HIT_VY_PER_LEVEL=8.0),
        "Daily Player": dict(START_SPEED=300.0, AI_BASE_SPEED=270.0,
                             MAX_BALL_SPEED=580.0, MAX_BALL_SPEED_PER_LEVEL=12.0,
                             HIT_SPEED_MULT=1.020, HIT_VY_BASE=260.0, HIT_VY_PER_LEVEL=10.0),
        "Expert": dict(START_SPEED=340.0, AI_BASE_SPEED=310.0,
                       MAX_BALL_SPEED=660.0, MAX_BALL_SPEED_PER_LEVEL=14.0,
                       HIT_SPEED_MULT=1.028, HIT_VY_BASE=300.0, HIT_VY_PER_LEVEL=12.0),
    }
    def current_difficulty():
        return difficulty_modes[difficulty_index]
    difficulty_ai_error_pix = 0
    difficulty_ai_err_min = 0.4
    difficulty_ai_err_max = 0.9
    def make_ball(direction=1, speed_scale=1.0, angle_offset=0.0):
        speed=START_SPEED*(1.0+0.08*(level-1))*speed_scale; ang=random.uniform(-0.35,0.35)+angle_offset
        vx=speed*direction*math.cos(ang); vy=speed*math.sin(ang); rect=pygame.Rect(WIDTH//2-BALL_SIZE//2, HEIGHT//2-BALL_SIZE//2, BALL_SIZE, BALL_SIZE)
        return {"rect":rect,"vx":vx,"vy":vy,"spin":0.0,"prev":rect.center,"last":None}

    
    def clamp_ball_speed(ball):
        max_speed = MAX_BALL_SPEED + MAX_BALL_SPEED_PER_LEVEL*level
        s = math.hypot(ball["vx"], ball["vy"])
        if s > max_speed and s > 0:
            scale = max_speed / s
            ball["vx"] *= scale
            ball["vy"] *= scale

        # --- Minimum speed floor to avoid stalls (e.g., after magnet ends)
        s = math.hypot(ball["vx"], ball["vy"])
        min_speed = max(80.0, MIN_BALL_SPEED + MIN_BALL_SPEED_PER_LEVEL*(level-1))
        try:
            if effects.get("slow_ball", 0.0) > 0.0:
                min_speed *= 0.45  # allow slowdown power-up to feel meaningful
        except Exception:
            pass
        if s < min_speed:
            if s <= 0.0001:
                # Re-kick the ball in the last known general direction
                direction = 1 if ball.get("vx", 1.0) >= 0 else -1
                if ball.get("last") == "ai":
                    direction = -1
                elif ball.get("last") == "player":
                    direction = 1
                else:
                    # Kick away from whichever side we're on
                    cx = ball["rect"].centerx if "rect" in ball else WIDTH//2
                    direction = 1 if cx < WIDTH//2 else -1
                ang = random.uniform(-0.28, 0.28)
                ball["vx"] = math.cos(ang) * min_speed * direction
                ball["vy"] = math.sin(ang) * min_speed
            else:
                scale = min_speed / max(s, 1e-6)
                ball["vx"] *= scale
                ball["vy"] *= scale
            s = min_speed

        # --- Avoid near-vertical traps: enforce a minimum horizontal component
        s = math.hypot(ball["vx"], ball["vy"]) or min_speed
        min_horiz = s * MIN_HORIZONTAL_RATIO
        if abs(ball["vx"]) < min_horiz:
            # Pick a sensible horizontal sign
            sign = 1 if ball.get("vx", 0.0) >= 0 else -1
            if abs(ball.get("vx", 0.0)) < 1e-6:
                if ball.get("last") == "ai":
                    sign = -1
                elif ball.get("last") == "player":
                    sign = 1
                else:
                    cx = ball["rect"].centerx if "rect" in ball else WIDTH//2
                    sign = 1 if cx < WIDTH//2 else -1
            ball["vx"] = min_horiz * sign
            # Adjust vy to preserve overall speed and keep current vertical direction if possible
            remaining = max(s*s - ball["vx"]*ball["vx"], 1.0)
            vy_sign = 1 if ball.get("vy", 0.0) >= 0 else -1
            if abs(ball.get("vy", 0.0)) < 1e-6:
                vy_sign = 1 if random.random() < 0.5 else -1
            ball["vy"] = math.sqrt(remaining) * vy_sign
    


    def apply_difficulty():
        """Apply current difficulty by changing tuning variables and nudging live speeds."""
        nonlocal ai_speed, difficulty_ai_error_pix, difficulty_ai_err_min, difficulty_ai_err_max
        settings = DIFFICULTY_SETTINGS[current_difficulty()]
        # Update global tuning vars
        globals()["START_SPEED"] = settings["START_SPEED"]
        globals()["AI_BASE_SPEED"] = settings["AI_BASE_SPEED"]
        globals()["MAX_BALL_SPEED"] = settings["MAX_BALL_SPEED"]
        globals()["MAX_BALL_SPEED_PER_LEVEL"] = settings["MAX_BALL_SPEED_PER_LEVEL"]
        globals()["HIT_SPEED_MULT"] = settings["HIT_SPEED_MULT"]
        globals()["HIT_VY_BASE"] = settings["HIT_VY_BASE"]
        globals()["HIT_VY_PER_LEVEL"] = settings["HIT_VY_PER_LEVEL"]
        # Update AI speed for current level
        ai_speed = AI_BASE_SPEED*(1.0+0.06*(level-1))
        # Nudge any active balls toward the new baseline & clamp
        try:
            for b in balls:
                s = (b["vx"]**2 + b["vy"]**2)**0.5
                target = START_SPEED*(1.0+0.08*(level-1))
                if s > 1 and target > 1:
                    scale = max(0.7, min(1.3, target / s))
                    b["vx"] *= scale; b["vy"] *= scale
                clamp_ball_speed(b)
        except Exception:
            pass


    player_trail=Trail(0); ai_trail=Trail(0); particles=[]; textfx=TextFX(mid); bigfx=BigPopupFX(big, small)

    player_score=0; ai_score=0; level=1; ai_speed=AI_BASE_SPEED; shake=0.0
    # Track goals since last level so level-ups are based ONLY on goals, not bonus points
    goals_since_level = 0
    powerup=None; next_powerup_time=random.uniform(POWERUP_COOLDOWN_MIN, POWERUP_COOLDOWN_MAX); elapsed=0.0
    # Always-on separate point token (respawns immediately on pickup)
    point_token=None
    paused=False; show_help=False; show_scores=False
    ai_aim_offset = 0.0
    ai_aim_timer = 0.0

    effects={"player_grow":0.0,"ai_grow":0.0,"player_shrink":0.0,"ai_shrink":0.0,"slow_ball":0.0,"player_magnet":0.0,"ai_magnet":0.0,"player_curve":0.0,"ai_curve":0.0,"player_shield":0.0,"ai_shield":0.0,"player_hardsmack":0.0,"ai_hardsmack":0.0,"player_triple":0.0,"ai_triple":0.0,"small_ball":0.0}
    original_sizes={"player_h":PADDLE_H,"ai_h":PADDLE_H}
    def apply_sizes():
        ph=original_sizes["player_h"]; ah=original_sizes["ai_h"]
        if effects["player_grow"]>0.0: ph=int(original_sizes["player_h"]*1.4)
        if effects["ai_grow"]>0.0: ah=int(original_sizes["ai_h"]*1.4)
        if effects["player_shrink"]>0.0: ph=int(original_sizes["player_h"]*0.65)
        if effects["ai_shrink"]>0.0: ah=int(original_sizes["ai_h"]*0.65)
        player.h=ph; ai.h=ah; player.y=max(10,min(HEIGHT-player.h-10,player.y)); ai.y=max(10,min(HEIGHT-ai.h-10,ai.y))
    def level_up():
        nonlocal level, ai_speed, player_score
        if level<MAX_LEVEL:
            level+=1; ai_speed=AI_BASE_SPEED*(1.0+0.06*(level-1))
            # +200 points on every level up
            player_score += 200
            textfx.add(f"Level {level}! +200", (WIDTH//2, TOPBAR_H+60), rainbow(level*0.2), 1.1, rise=-60)

    def emit_spark(x,y,color,count=22):
        for _ in range(count): particles.append(Particle(x,y,color))

    def start_game():
        nonlocal player_score, ai_score, level, ai_speed, shake, state, balls, powerup, next_powerup_time, elapsed, effects, paused, show_help, show_scores, goals_since_level, point_token
        player_score=0; ai_score=0; level=1; ai_speed=AI_BASE_SPEED; shake=0.0; apply_difficulty()
        goals_since_level = 0
        balls=[make_ball(direction=random.choice([-1,1]))]; powerup=None; next_powerup_time=random.uniform(POWERUP_COOLDOWN_MIN, POWERUP_COOLDOWN_MAX); elapsed=0.0
        # Spawn a point token immediately
        point_token = None
        spawn_point_token()  # will set point_token
        for k in list(effects.keys()): effects[k]=0.0
        apply_sizes()
        for _ in range(90): emit_spark(WIDTH//2, HEIGHT//2, rainbow(random.random()*6), 1)
        state="PLAY"; paused=False; show_help=False; show_scores=False

    def finish_game():
        nonlocal state
        state="GAME_OVER"
        scores["best_score"]=max(scores.get("best_score",0), player_score)
        scores["best_level"]=max(scores.get("best_level",1), level)
        scores["total_played"]=scores.get("total_played",0)+1
        save_scores(scores)

    class PowerUp:
        COLORS={"GROW":(100,255,120),"SHRINK_OPP":(255,120,120),"SLOW_BALL":(120,180,255),"MULTI":(255,210,120),"CURVE":(255,180,255),"SHIELD":(180,255,255),"HARD_SMACK":(255,230,120),"TRIPLE_SIZE":(200,160,255),"SMALL_BALL":(180,200,255)}
        def __init__(self, kind, x, y):
            self.kind=kind; self.rect=pygame.Rect(x,y,POWERUP_SIZE,POWERUP_SIZE); self.t=0.0; self.vy=random.uniform(-30,30); self.color=self.COLORS[kind]
        def update(self, dt, balls):
            self.t+=dt; self.rect.y+=int(self.vy*dt)
            if self.rect.top<=6 or self.rect.bottom>=HEIGHT-6: self.vy*=-1
            if balls:
                bx,by=min([(b['rect'].centerx,b['rect'].centery) for b in balls], key=lambda p:(p[0]-self.rect.centerx)**2+(p[1]-self.rect.centery)**2)
                dx,dy=bx-self.rect.centerx, by-self.rect.centery; d2=dx*dx+dy*dy
                if d2<=POWERUP_ATTRACT_RADIUS*POWERUP_ATTRACT_RADIUS:
                    d=max(1.0, math.sqrt(d2)); self.rect.centerx+=int((dx/d)*POWERUP_ATTRACT_SPEED*dt); self.rect.centery+=int((dy/d)*POWERUP_ATTRACT_SPEED*dt)
        def draw(self, surf, phase):
            glow=soft_glow_surf(24,self.color,140); surf.blit(glow, glow.get_rect(center=self.rect.center), special_flags=pygame.BLEND_ADD)
            pygame.draw.circle(surf, (*self.color,230), self.rect.center, POWERUP_SIZE//2); pygame.draw.circle(surf,(255,255,255,200), self.rect.center, POWERUP_SIZE//2-4,2)
            cx,cy=self.rect.center
            if self.kind=="GROW": pygame.draw.polygon(surf,(25,25,25),[(cx-6,cy+6),(cx,cy-8),(cx+6,cy+6)],0)
            elif self.kind=="SHRINK_OPP": pygame.draw.rect(surf,(25,25,25),(cx-10,cy-3,20,6), border_radius=3)
            elif self.kind=="SLOW_BALL": pygame.draw.circle(surf,(25,25,25),(cx,cy),5)
            elif self.kind=="MULTI": pygame.draw.circle(surf,(25,25,25),(cx-4,cy),3); pygame.draw.circle(surf,(25,25,25),(cx+4,cy),3)
            elif self.kind=="HARD_SMACK": pygame.draw.polygon(surf,(25,25,25),[(cx-2,cy-12),(cx+4,cy-12),(cx,cy-2),(cx+6,cy-2),(cx-2,cy+12),(cx+2,cy+2),(cx-4,cy+2)],0)
            elif self.kind=="TRIPLE_SIZE": pygame.draw.rect(surf,(25,25,25),(cx-5,cy-12,10,24),2,border_radius=3); pygame.draw.line(surf,(25,25,25),(cx-9,cy),(cx-13,cy),2); pygame.draw.line(surf,(25,25,25),(cx+9,cy),(cx+13,cy),2)
            elif self.kind=="SMALL_BALL": pygame.draw.circle(surf,(25,25,25),(cx,cy),3)
            elif self.kind=="CURVE": pygame.draw.arc(surf,(25,25,25),(cx-10,cy-10,20,20),0.3,2.6,3)
            elif self.kind=="SHIELD": pygame.draw.rect(surf,(25,25,25),(cx-8,cy-10,16,20),2)

    # --- Separate point token (1 / 5 / 10) that always respawns ---
    class PointToken:
        COLORS = {
            1: (90,150,255),    # blue
            5: (255,160,70),    # orange
            10: (100,225,120),  # green
        }
        SIZE = 22
        def __init__(self, value, x, y):
            self.value = value
            self.color = self.COLORS[value]
            self.rect = pygame.Rect(x, y, self.SIZE, self.SIZE)
            self.vx = random.uniform(-40, 40)
            self.vy = random.uniform(-40, 40)
        def update(self, dt):
            self.rect.x += int(self.vx * dt)
            self.rect.y += int(self.vy * dt)
            if self.rect.left <= 6 or self.rect.right >= WIDTH-6: self.vx *= -1
            if self.rect.top <= 6 or self.rect.bottom >= HEIGHT-6: self.vy *= -1
        def draw(self, surf, font):
            glow = soft_glow_surf(18, self.color, 140)
            surf.blit(glow, glow.get_rect(center=self.rect.center), special_flags=pygame.BLEND_ADD)
            pygame.draw.circle(surf, (*self.color, 230), self.rect.center, self.SIZE//2)
            pygame.draw.circle(surf, (255,255,255,210), self.rect.center, self.SIZE//2-3, 2)
            label = str(self.value)
            txt = font.render(label, True, (20,20,20))
            surf.blit(txt, txt.get_rect(center=self.rect.center))

    def spawn_powerup():
        nonlocal powerup
        kind=random.choice(POWERUP_TYPES); margin=80
        x=random.randint(WIDTH//3, WIDTH*2//3-POWERUP_SIZE); y=random.randint(margin, HEIGHT-margin-POWERUP_SIZE)
        powerup=PowerUp(kind, x, y)

    def spawn_point_token():
        """Respawn the separate point token immediately at a random spot."""
        nonlocal point_token
        # Weighted choice to make 1pt most common, then 5, then 10
        values = [1]*6 + [5]*3 + [10]*1
        value = random.choice(values)
        margin = 60
        x = random.randint(margin, WIDTH-margin-PointToken.SIZE)
        y = random.randint(margin, HEIGHT-margin-PointToken.SIZE)
        point_token = PointToken(value, x, y)

    def announce_pickup(kind, owner):
        color=PowerUp.COLORS.get(kind,(255,255,255)); label=kind.title().replace("_"," "); explain=POWERUP_EXPLAINS.get(kind,"")
        x=int(WIDTH*0.30) if owner=="player" else int(WIDTH*0.70)
        bigfx.add(label, explain, (x, TOPBAR_H+90), color, ttl=1.25)
        for _ in range(26): particles.append(Particle(x, TOPBAR_H+90, color))

    def award(kind, owner):
        if owner=="player":
            if kind=="GROW": effects["player_grow"]=6.0
            elif kind=="SHRINK_OPP": effects["ai_shrink"]=6.0
            elif kind=="SLOW_BALL": effects["slow_ball"]=5.0
            elif kind=="MULTI":
                if balls:
                    direction=1 if balls[0]["vx"]>=0 else -1
                    # add two extra balls for 3 total
                    for offs in (-0.25, 0.25):
                        angle_offset=random.uniform(-0.35,0.35)+offs
                        balls.append(make_ball(direction=direction, angle_offset=angle_offset))
            elif kind=="CURVE": effects["player_curve"]=8.0
            elif kind=="SHIELD": effects["player_shield"]=8.0
            elif kind=="HARD_SMACK": effects["player_hardsmack"]=10.0
            elif kind=="TRIPLE_SIZE": effects["player_triple"]=10.0
            elif kind=="SMALL_BALL": effects["small_ball"]=10.0
            textfx.add(labelify(kind), (WIDTH*0.33, TOPBAR_H+56), (255,255,255), 0.9, rise=-50); announce_pickup(kind,"player")
        else:
            if kind=="GROW": effects["ai_grow"]=6.0
            elif kind=="SHRINK_OPP": effects["player_shrink"]=6.0
            elif kind=="SLOW_BALL": effects["slow_ball"]=5.0
            elif kind=="MULTI":
                if balls:
                    direction=-1 if balls[0]["vx"]>=0 else 1
                    for offs in (-0.25, 0.25):
                        angle_offset=random.uniform(-0.35,0.35)+offs
                        balls.append(make_ball(direction=direction, angle_offset=angle_offset))
            elif kind=="CURVE": effects["ai_curve"]=8.0
            elif kind=="SHIELD": effects["ai_shield"]=8.0
            elif kind=="HARD_SMACK": effects["ai_hardsmack"]=10.0
            elif kind=="TRIPLE_SIZE": effects["ai_triple"]=10.0
            elif kind=="SMALL_BALL": effects["small_ball"]=10.0
            textfx.add("AI "+labelify(kind), (WIDTH*0.67, TOPBAR_H+56), (255,230,230), 0.9, rise=-50); announce_pickup(kind,"ai")
    def labelify(k): return k.title().replace("_"," ")

    clock=pygame.time.Clock(); running=True; button_rects={}
    while running:
        dt=clock.tick(FPS)/1000.0; view_rect=compute_view_rect(); bg_phase+=dt*(0.35+0.05*level)

        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_F11: toggle_maximize()
                if event.key==pygame.K_ESCAPE:
                    if show_help or show_scores:
                        show_help=False; show_scores=False
                        if state=="PLAY": paused=False
                    else:
                        if state=="PLAY": finish_game()
                if state=="MENU":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE): start_game()
                elif state=="PLAY":
                    if event.key==pygame.K_p and not (show_help or show_scores): paused=not paused
                elif state=="GAME_OVER":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE): start_game()
            elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                pos=window_to_canvas(event.pos, view_rect); now=pygame.time.get_ticks()
                if pos is not None:
                    x,y=pos
                    if y<=TOPBAR_H+8:
                        if now - last_click_time <= DOUBLE_CLICK_MS: toggle_maximize()
                        last_click_time=now
                        for name, rect in button_rects.items():
                            if rect.collidepoint((x,y)):
                                if name=="PAUSE":
                                    if state=="PLAY": paused=not paused; show_help=False; show_scores=False
                                elif name=="HELP":
                                    show_help=not show_help; show_scores=False
                                    if state=="PLAY": paused = show_help or paused
                                elif name=="SCORES":
                                    show_scores=not show_scores; show_help=False
                                    if state=="PLAY": paused = show_scores or paused

                                elif name=="DIFF":
                                    difficulty_index = (difficulty_index + 1) % len(difficulty_modes)
                                    apply_difficulty()
                                    textfx.add("Mode: "+current_difficulty(), (WIDTH*0.5, TOPBAR_H+54), (255,255,255), 0.9, rise=-40)
                                elif name=="EXIT":
                                    if state=="PLAY": finish_game()
                                    else: running=False

        # Background
        canvas.fill((0,0,0,0)); grad=pygame.Surface((WIDTH,HEIGHT))
        for y in range(0,HEIGHT,6): pygame.draw.rect(grad, rainbow((bg_phase*0.2 + y/HEIGHT)*1.5), (0,y,WIDTH,6))
        canvas.blit(grad,(0,0))
        for y in range(0,HEIGHT,28): pygame.draw.rect(canvas,(255,255,255,120),(WIDTH//2-3,y,6,18))

        # Semi-transparent background image overlay (fills screen)
        if bg_images:
            canvas.blit(bg_images[bg_index], (0, 0))
        keys=pygame.key.get_pressed()

        # Always show top bar (ALL states)
        button_rects={}; bar=pygame.Surface((WIDTH,TOPBAR_H), pygame.SRCALPHA); bar.fill((0,0,0,110)); canvas.blit(bar,(0,0))
        labels=[("PAUSE","Resume" if (state=="PLAY" and paused and not (show_help or show_scores)) else "Pause"),
                ("HELP","Help"),("SCORES","High Scores"),("DIFF","Difficulty: "+current_difficulty()),("EXIT","Exit")]
        mx_my=window_to_canvas(pygame.mouse.get_pos(), view_rect); mx,my=(-1,-1) if mx_my is None else mx_my
        xbtn=10
        for key,label in labels:
            w,h=small.size(label); r=pygame.Rect(xbtn,4,w+24,TOPBAR_H-8); hot=r.collidepoint((mx,my))
            UIButton(label,r).draw(canvas,small,hot=hot); button_rects[key]=r; xbtn+=r.w+8

        if state=="MENU":
            title=big.render("Colorful Pong",True,rainbow(bg_phase)); canvas.blit(title,title.get_rect(center=(WIDTH//2,HEIGHT//2-60)))
            txt=mid.render("Press ENTER to Start",True,(240,240,240)); canvas.blit(txt,txt.get_rect(center=(WIDTH//2,HEIGHT//2+16)))
            hs=small.render(f"Best Score: {scores.get('best_score',0)}   Best Level: {scores.get('best_level',1)}   Played: {scores.get('total_played',0)}", True,(250,250,250))
            canvas.blit(hs,hs.get_rect(center=(WIDTH//2, HEIGHT-32)))
        if state in ("PLAY","GAME_OVER"):
            do_update=(state=="PLAY") and (not paused) and (not show_help) and (not show_scores)
            if state=="PLAY" and do_update:
                move=0
                if keys[pygame.K_w] or keys[pygame.K_UP]: move-=1
                if keys[pygame.K_s] or keys[pygame.K_DOWN]: move+=1
                player.y+=int(move*420*dt); player.y=max(10,min(HEIGHT-player.h-10,player.y))
                # Horizontal movement (player can move left/right up to the middle)
                move_x = 0
                if keys[pygame.K_a] or keys[pygame.K_LEFT]: move_x -= 1
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move_x += 1
                player.x += int(move_x * 420 * dt)
                min_x = 10
                # Keep a small margin from the dashed midline (which is 6px wide centered at WIDTH//2)
                max_x = (WIDTH//2 - 8) - player.w
                if player.x < min_x: player.x = min_x
                if player.x > max_x: player.x = max_x
                target_y=min(balls,key=lambda b:abs(b["rect"].centerx-ai.centerx))["rect"].centery if balls else HEIGHT//2
                ai_aim_timer -= dt
                if ai_aim_timer <= 0.0:
                    # Randomize AI aim offset based on difficulty
                    ai_aim_offset = random.uniform(-difficulty_ai_error_pix, difficulty_ai_error_pix)
                    import random as _r
                    ai_aim_timer = _r.uniform(difficulty_ai_err_min, difficulty_ai_err_max)
                target_y += ai_aim_offset
                dy=target_y-ai.centery; ai.y+=int(max(-1,min(1,dy*AI_REACTION))*ai_speed*dt); ai.y=max(10,min(HEIGHT-ai.h-10,ai.y))
                if powerup is None and elapsed>=next_powerup_time: spawn_powerup()
                if powerup: powerup.update(dt, balls)
                # Always ensure a point token exists
                if point_token is None: spawn_point_token()
                else: point_token.update(dt)
                for k in list(effects.keys()):
                    if effects[k]>0.0: effects[k]=max(0.0, effects[k]-dt)
                apply_sizes()
            left_shield=pygame.Rect(6,0,8,HEIGHT) if effects["player_shield"]>0.0 else None
            right_shield=pygame.Rect(WIDTH-14,0,8,HEIGHT) if effects["ai_shield"]>0.0 else None
            slow_factor=0.7 if effects["slow_ball"]>0.0 else 1.0; magnet_accel=220.0
            for b in balls:
                bx,by=b["rect"].center
                # SMALL_BALL effect: balls become half size while active
                if effects.get("small_ball",0.0)>0.0:
                    target = max(4, BALL_SIZE//2)
                else:
                    target = BALL_SIZE
                if b["rect"].w != target:
                    cx,cy=b["rect"].center
                    b["rect"].w = b["rect"].h = target
                    b["rect"].center=(cx,cy)
                if state=="PLAY" and do_update:
                    b["prev"]=(bx,by)
                    if effects["player_magnet"]>0.0:
                        dx=player.centerx-bx; dyb=player.centery-by; d=max(1.0,math.hypot(dx,dyb))
                        b["vx"]+=(dx/d)*magnet_accel*dt; b["vy"]+=(dyb/d)*magnet_accel*dt
                    if effects["ai_magnet"]>0.0:
                        dx=ai.centerx-bx; dyb=ai.centery-by; d=max(1.0,math.hypot(dx,dyb))
                        b["vx"]+=(dx/d)*magnet_accel*dt; b["vy"]+=(dyb/d)*magnet_accel*dt
                    if abs(b.get("spin",0.0))>0.1: b["vy"]+=b["spin"]*dt; b["spin"]*=0.985
                    b["rect"].x+=int(b["vx"]*slow_factor*dt); b["rect"].y+=int(b["vy"]*slow_factor*dt); clamp_ball_speed(b)
                    if left_shield and b["vx"]<0 and b["rect"].colliderect(left_shield): b["rect"].left=left_shield.right; b["vx"]*=-1
                    if right_shield and b["vx"]>0 and b["rect"].colliderect(right_shield): b["rect"].right=right_shield.left; b["vx"]*=-1
                    if b["rect"].top<=0: b["rect"].top=0; b["vy"]*=-1
                    if b["rect"].bottom>=HEIGHT: b["rect"].bottom=HEIGHT; b["vy"]*=-1
                    if b["rect"].colliderect(player):
                        offset=(b["rect"].centery-player.centery)/(player.h/2)
                        sm_mult = 1.18 if effects.get("player_hardsmack",0.0)>0.0 else 1.0
                        b["vx"]=abs(b["vx"])*(HIT_SPEED_MULT*sm_mult); vy_term=(HIT_VY_BASE+HIT_VY_PER_LEVEL*level)*sm_mult
                        b["vy"]=(b["vy"]*0.6)+offset*vy_term; b["last"]="player"; clamp_ball_speed(b)
                        if effects["player_curve"]>0.0: b["spin"]=280.0*offset; textfx.add("Curve!", (WIDTH*0.35, TOPBAR_H+74), (255,220,255), 0.6, rise=-40)
                    elif b["vx"]>0 and b["rect"].colliderect(ai):
                        offset=(b["rect"].centery-ai.centery)/(ai.h/2)
                        sm_mult = 1.18 if effects.get("ai_hardsmack",0.0)>0.0 else 1.0
                        b["vx"]=-abs(b["vx"])*(HIT_SPEED_MULT*sm_mult); vy_term=(HIT_VY_BASE+HIT_VY_PER_LEVEL*level)*sm_mult
                        b["vy"]=(b["vy"]*0.6)+offset*vy_term; b["last"]="ai"; clamp_ball_speed(b)
                        if effects["ai_curve"]>0.0: b["spin"]=280.0*offset
            # Power-up pickup (normal power-ups)
            if state=="PLAY" and do_update and powerup:
                pick_rect=powerup.rect.inflate(POWERUP_PICKUP_PAD, POWERUP_PICKUP_PAD); collected=False; owner=None
                for b in balls:
                    if b["rect"].colliderect(pick_rect) or segment_intersects_rect(b["prev"], b["rect"].center, pick_rect):
                        collected=True; owner=b["last"] or ("player" if b["vx"]<0 else "ai"); break
                    bx,by=b["rect"].center; cx,cy=powerup.rect.center
                    if (bx-cx)**2+(by-cy)**2 <= POWERUP_SIZE**2: collected=True; owner=b["last"] or ("player" if b["vx"]<0 else "ai"); break
                if collected:
                    kind=powerup.kind; award(kind, owner)
                    for _ in range(36): particles.append(Particle(powerup.rect.centerx, powerup.rect.centery, powerup.color))
                    powerup=None; elapsed=0.0; next_powerup_time=random.uniform(POWERUP_COOLDOWN_MIN, POWERUP_COOLDOWN_MAX)
            # Point token pickup (separate, immediate respawn)
            if state=="PLAY" and do_update and point_token:
                pick_rect = point_token.rect.inflate(POWERUP_PICKUP_PAD, POWERUP_PICKUP_PAD)
                collected=False; owner=None
                for b in balls:
                    if b["rect"].colliderect(pick_rect) or segment_intersects_rect(b["prev"], b["rect"].center, pick_rect):
                        collected=True; owner=b["last"] or ("player" if b["vx"]<0 else "ai"); break
                if collected:
                    val = point_token.value
                    if owner=="player":
                        player_score += val
                        textfx.add(f"+{val}", (WIDTH*0.33, TOPBAR_H+72), (255,255,255), 0.9, rise=-48)
                    else:
                        ai_score += val
                        textfx.add(f"AI +{val}", (WIDTH*0.67, TOPBAR_H+72), (255,230,230), 0.9, rise=-48)
                    for _ in range(24): particles.append(Particle(point_token.rect.centerx, point_token.rect.centery, point_token.color))
                    spawn_point_token()
            if state=="PLAY" and do_update:
                # Scoring & level logic
                scored_left=0; scored_right=0; new_balls=[]
                for b in balls:
                    if b["rect"].right<0: scored_right+=1; continue
                    elif b["rect"].left>WIDTH: scored_left+=1; continue
                    new_balls.append(b)
                balls=new_balls
                if scored_right>0: ai_score+=scored_right
                if scored_left>0:
                    player_score+=scored_left
                    goals_since_level += scored_left
                # Level-ups are based on number of PLAYER GOALS only
                while goals_since_level >= LEVEL_UP_EVERY:
                    level_up()
                    goals_since_level -= LEVEL_UP_EVERY
                # Change background image index on player scores (if available)
                if bg_images:
                    bg_index = (bg_index + scored_left) % len(bg_images)
                # Ensure at least one ball remains
                if not balls: balls.append(make_ball(direction=-1 if scored_left>0 else 1))
                # Game over if trailing by >10
                if (ai_score - player_score) > 10:
                    finish_game()
                elapsed+=dt
            if state=="PLAY" and do_update:
                particles[:]=[p for p in particles if p.update(dt)]; textfx.update(dt); bigfx.update(dt)
            player_trail.add(player, rainbow(bg_phase*0.7)); ai_trail.add(ai, rainbow(bg_phase*0.7+0.5))
            player_trail.draw(canvas); ai_trail.draw(canvas); pygame.draw.rect(canvas,(240,240,240),player, border_radius=8)
            pygame.draw.rect(canvas,(240,240,240),ai, border_radius=8)
            for b in balls:
                canvas.blit(ball_glow, (b["rect"].centerx-48, b["rect"].centery-48), special_flags=pygame.BLEND_ADD)
                pygame.draw.rect(canvas,(250,250,250), b["rect"], border_radius=6)
            if powerup: powerup.draw(canvas, bg_phase)
            if point_token: point_token.draw(canvas, small)
            if effects["player_shield"]>0.0: s=pygame.Surface((10,HEIGHT), pygame.SRCALPHA); s.fill((180,255,255,120)); canvas.blit(s,(6,0))
            if effects["ai_shield"]>0.0: s=pygame.Surface((10,HEIGHT), pygame.SRCALPHA); s.fill((180,255,255,120)); canvas.blit(s,(WIDTH-16,0))
            hud=big.render(f"{player_score} : {ai_score}", True,(255,255,255)); canvas.blit(hud, hud.get_rect(center=(WIDTH//2, TOPBAR_H+24)))
            lvl=small.render(f"Level {level}", True,(255,255,255)); canvas.blit(lvl,(12, TOPBAR_H+10))

        if state=="GAME_OVER":
            over=big.render("Game Over",True,rainbow(bg_phase)); canvas.blit(over, over.get_rect(center=(WIDTH//2,HEIGHT//2-60)))
            results=mid.render(f"Score {player_score}  •  Level {level}",True,(250,250,250)); canvas.blit(results, results.get_rect(center=(WIDTH//2,HEIGHT//2+6)))
            info=small.render("Enter: play again   Esc: menu",True,(240,240,240)); canvas.blit(info, info.get_rect(center=(WIDTH//2,HEIGHT//2+48)))
            hs=small.render(f"Best Score: {scores.get('best_score',0)}   Best Level: {scores.get('best_level',1)}   Played: {scores.get('total_played',0)}", True,(250,250,250))
            canvas.blit(hs, hs.get_rect(center=(WIDTH//2, HEIGHT-32)))

        if show_help:
            dim=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA); dim.fill((0,0,0,140)); canvas.blit(dim,(0,0))
            panel=pygame.Surface((int(WIDTH*0.86), int(HEIGHT*0.7)), pygame.SRCALPHA); panel.fill((25,25,25,200))
            rx,ry=(WIDTH-panel.get_width())//2, (HEIGHT-panel.get_height())//2; canvas.blit(panel,(rx,ry))
            title=mid.render("How to Play",True,(255,255,255)); canvas.blit(title,(rx+18,ry+12))
            lines=["Goal: Hit the ball past the opponent. Power-ups spawn in the middle.","Last player to hit the ball 'owns' the next power-up collect.","","Controls:","  W / S   or   Up / Down  — move","  P                      — pause/resume","  F11 or Double-click top bar — maximize window","  Esc                    — close overlay / end run (if playing)","","Power-Ups:"]
            y=ry+54
            for line in lines:
                r=small.render(line,True,(235,235,235)); canvas.blit(r,(rx+18,y)); y+=24
            for k in POWERUP_TYPES:
                c=PowerUp.COLORS[k]; pygame.draw.circle(canvas,c,(rx+28,y+8),6)
                r=small.render(f"{k.title().replace('_',' ')} — {POWERUP_EXPLAINS.get(k,'')}", True,(235,235,235)); canvas.blit(r,(rx+44,y)); y+=22
            # Token explainer
            y+=10
            r=small.render("Point Tokens: Blue=+1, Orange=+5, Green=+10 (always respawn)", True,(235,235,235)); canvas.blit(r,(rx+18,y))
        if show_scores:
            dim=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA); dim.fill((0,0,0,140)); canvas.blit(dim,(0,0))
            panel=pygame.Surface((int(WIDTH*0.5), int(HEIGHT*0.4)), pygame.SRCALPHA); panel.fill((25,25,25,200))
            rx,ry=(WIDTH-panel.get_width())//2, (HEIGHT-panel.get_height())//2; canvas.blit(panel,(rx,ry))
            title=mid.render("High Scores",True,(255,255,255)); canvas.blit(title,(rx+18,ry+12))
            s1=small.render(f"Best Score:  {scores.get('best_score',0)}", True,(235,235,235)); s2=small.render(f"Best Level:  {scores.get('best_level',1)}", True,(235,235,235))
            s3=small.render(f"Games Played: {scores.get('total_played',0)}", True,(235,235,235))
            canvas.blit(s1,(rx+18,ry+56)); canvas.blit(s2,(rx+18,ry+80)); canvas.blit(s3,(rx+18,ry+104))

        textfx.draw(canvas); bigfx.draw(canvas)
        scaled=pygame.transform.smoothscale(canvas,(view_rect.w, view_rect.h)); display.fill((0,0,0)); display.blit(scaled, view_rect.topleft); pygame.display.flip()

    pygame.quit()

if __name__=="__main__": main()
