#!/usr/bin/env python3
import os, sys, json, random, math, datetime, re
import pygame

# ------------------------------ Paths & Files ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

BG_DIR = os.path.join(ROOT_DIR, 'presets', 'startup')
ASSET_DIR = os.path.join(ROOT_DIR, 'assets', 'frameshooter')
SAVE_DIR = os.path.join(ROOT_DIR, 'presets', 'setsave', 'frameshooters')
HS_FILE = os.path.join(SAVE_DIR, 'highscores.json')
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------ Window & DPI -------------------------------
CAPTION = "FrameShooters — retro sky shooter"
BASE_W, BASE_H = 1280, 720
TOPBAR_H = 44
BG_OVERLAY_ALPHA = int(255 * 0.20)

# ------------------------------ Constants ----------------------------------
POWERUP_FILES = {
    "spread": "powerup_spread.png",
    "rapid": "powerup_rapid.png",
    "shield": "powerup_shield.png",
    "bomb": "powerup_bomb.png",
    "magnet": "powerup_magnet.png",
}
POWERUP_MAX_SIZE = 36  # px bounding box for auto-resize, keeping aspect
POWERUP_DURATIONS = {
    "spread": 8.0,
    "rapid": 6.0,
    "shield": 5.0,
    "magnet": 8.0,
    # "bomb" is instant, no timer
}

COMBO_MAX_LEVEL = 5
COMBO_STEP_TIME = 2.0  # time to hold each step before decay

# ------------------------------ Helpers -----------------------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def load_images_from(folder, exts=('.png', '.jpg', '.jpeg')):
    imgs = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(exts):
                path = os.path.join(folder, name)
                try:
                    img = pygame.image.load(path).convert_alpha()
                    imgs.append(img)
                except Exception:
                    pass
    return imgs

def scale_to_box(img, max_size):
    w, h = img.get_size()
    if w <= max_size and h <= max_size:
        return img
    if w >= h:
        new_w = max_size
        new_h = max(1, int(h * (max_size / w)))
    else:
        new_h = max_size
        new_w = max(1, int(w * (max_size / h)))
    return pygame.transform.smoothscale(img, (new_w, new_h))

def procedural_plane(color=(230,230,255), w=64, h=64):
    surf = pygame.Surface((w,h), pygame.SRCALPHA)
    pygame.draw.polygon(surf, color, [(w*0.5,0),(w*0.85,h*0.85),(w*0.15,h*0.85)])
    pygame.draw.line(surf, (100,100,180), (w*0.5, h*0.1), (w*0.5, h*0.7), 3)
    return surf

def procedural_enemy(color=(255,200,160), w=48, h=48):
    surf = pygame.Surface((w,h), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (6,6,w-12,h-12), border_radius=10)
    pygame.draw.circle(surf, (180,80,60), (w//2, h//2), min(w,h)//5)
    return surf

def procedural_boss(color=(255,120,120), w=160, h=120):
    surf = pygame.Surface((w,h), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (4,14,w-8,h-28), border_radius=16)
    pygame.draw.rect(surf, (120,30,30), (w*0.3, h*0.35, w*0.4, h*0.3), border_radius=10)
    pygame.draw.circle(surf, (220,50,50), (int(w*0.2), int(h*0.5)), 14)
    pygame.draw.circle(surf, (220,50,50), (int(w*0.8), int(h*0.5)), 14)
    return surf

def load_game_assets():
    """Load assets following explicit filenames. Falls back to procedural sprites."""
    def try_load(name):
        try:
            return pygame.image.load(os.path.join(ASSET_DIR, name)).convert_alpha()
        except Exception:
            return None

    player_img = try_load('player.png')

    enemy_imgs = []
    if os.path.isdir(ASSET_DIR):
        for fname in os.listdir(ASSET_DIR):
            m = re.match(r'enemy(\d+)\.png$', fname, re.IGNORECASE)
            if m:
                enemy_imgs.append((int(m.group(1)), fname))
        enemy_imgs.sort(key=lambda t: t[0])
    enemy_imgs = [try_load(fname) for _, fname in enemy_imgs]
    enemy_imgs = [img for img in enemy_imgs if img is not None]

    boss_imgs = []
    if os.path.isdir(ASSET_DIR):
        for fname in os.listdir(ASSET_DIR):
            m = re.match(r'boss(\d+)\.png$', fname, re.IGNORECASE)
            if m:
                boss_imgs.append((int(m.group(1)), fname))
        boss_imgs.sort(key=lambda t: t[0])
    boss_imgs = [try_load(fname) for _, fname in boss_imgs]
    boss_imgs = [img for img in boss_imgs if img is not None]

    # Power-ups
    powerup_imgs = {}
    for key, fname in POWERUP_FILES.items():
        img = try_load(fname)
        if img is not None:
            powerup_imgs[key] = scale_to_box(img, POWERUP_MAX_SIZE)

    if player_img is None:
        player_img = procedural_plane()
    if not enemy_imgs:
        enemy_imgs = [procedural_enemy()]
    if not boss_imgs:
        boss_imgs = [procedural_boss()]

    return player_img, enemy_imgs, boss_imgs, powerup_imgs

# ------------------------------ UI ----------------------------------------
class Button:
    def __init__(self, text, rect, callback):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.callback = callback
        self.hover = False

    def draw(self, surf, font):
        bg = (30,30,30) if not self.hover else (60,60,60)
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, (100,100,100), self.rect, 1, border_radius=8)
        label = font.render(self.text, True, (240,240,240))
        surf.blit(label, label.get_rect(center=self.rect.center))

    def handle_event(self, e):
        if e.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(e.pos)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.rect.collidepoint(e.pos):
                self.callback()

# ------------------------------ Entities ----------------------------------
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, dy, color=(255,255,100), radius=4, damage=1, vx=0):
        super().__init__()
        self.dy = dy
        self.vx = vx
        self.damage = damage
        self.image = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (radius, radius), radius)
        self.rect = self.image.get_rect(center=(x,y))

    def update(self, dt, bounds):
        self.rect.y += int(self.dy * dt)
        self.rect.x += int(self.vx * dt)
        if not bounds.colliderect(self.rect):
            self.kill()


class HomingBomb(pygame.sprite.Sprite):
    """A slow homing bomb that chases the player for a short time."""
    def __init__(self, x, y, target, speed=300, lifetime=5.0, radius=10):
        super().__init__()
        self.target = target
        self.speed = speed
        self.lifetime = lifetime
        self.image = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        # simple ring with inner fill to stand out
        pygame.draw.circle(self.image, (255, 180, 80), (radius, radius), radius)
        pygame.draw.circle(self.image, (30, 30, 30), (radius, radius), max(1, radius-4))
        self.rect = self.image.get_rect(center=(x, y))
        self._radius = radius

    def update(self, dt, bounds):
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.kill()
            return
        # steer towards player
        if self.target is not None:
            tx, ty = self.target.rect.centerx, self.target.rect.centery
            dx, dy = (tx - self.rect.centerx), (ty - self.rect.centery)
            dist = max(1.0, (dx*dx + dy*dy) ** 0.5)
            vx = (dx / dist) * self.speed
            vy = (dy / dist) * self.speed
            self.rect.x += int(vx * dt)
            self.rect.y += int(vy * dt)
        # keep on screen
        if not bounds.inflate(40, 40).colliderect(self.rect):
            # if goes too far out, remove
            self.kill()

class BossLaser(pygame.sprite.Sprite):
    """A big vertical laser that persists for a duration and hurts on touch."""
    def __init__(self, boss, width=28, duration=2.5):
        super().__init__()
        self.boss = boss
        self.width = int(width)
        self.duration = duration
        self.timer = duration
        self.image = pygame.Surface((self.width, 1), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self._rebuild_image(1)

    def _rebuild_image(self, height):
        # create a tall translucent beam
        surf = pygame.Surface((self.width, height), pygame.SRCALPHA)
        # core
        pygame.draw.rect(surf, (255, 80, 80, 180), (0, 0, self.width, height))
        # glow edges
        pygame.draw.rect(surf, (255, 160, 160, 90), (0, 0, max(1, self.width//4), height))
        pygame.draw.rect(surf, (255, 160, 160, 90), (self.width - max(1, self.width//4), 0, max(1, self.width//4), height))
        self.image = surf
        self.rect = self.image.get_rect()

    def update(self, dt, bounds):
        self.timer -= dt
        if self.timer <= 0:
            self.kill()
            return
        # follow boss x
        if self.boss and self.boss.alive():
            bx = self.boss.rect.centerx
            top = self.boss.rect.bottom - 4
        else:
            bx = bounds.centerx
            top = bounds.top + 80
        # beam covers from just under boss to bottom
        height = max(1, bounds.bottom - top)
        if self.image.get_height() != height or self.image.get_width() != self.width:
            self._rebuild_image(height)
        self.rect.topleft = (bx - self.width//2, top)

class PlayerLaser(pygame.sprite.Sprite):
    """Player's upward laser beam that persists briefly and deals damage over time."""
    def __init__(self, player, width=12, duration=0.35, dps=110):
        super().__init__()
        self.player = player
        self.width = int(width)
        self.duration = duration
        self.timer = duration
        self.dps = float(dps)
        self.image = pygame.Surface((self.width, 1), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self._rebuild_image(1)

    def _rebuild_image(self, height):
        surf = pygame.Surface((self.width, height), pygame.SRCALPHA)
        pygame.draw.rect(surf, (180, 220, 255, 170), (0, 0, self.width, height))
        edge = max(1, self.width//5)
        pygame.draw.rect(surf, (140, 200, 255, 110), (0, 0, edge, height))
        pygame.draw.rect(surf, (140, 200, 255, 110), (self.width-edge, 0, edge, height))
        self.image = surf
        self.rect = self.image.get_rect()

    def update(self, dt, bounds):
        self.timer -= dt
        if self.timer <= 0:
            self.kill()
            return
        if self.player and self.player.alive():
            px, py = self.player.rect.centerx, self.player.rect.top + 2
        else:
            px, py = bounds.centerx, bounds.bottom - 100
        height = max(1, py - bounds.top)
        if self.image.get_height() != height or self.image.get_width() != self.width:
            self._rebuild_image(height)
        self.rect.topleft = (px - self.width//2, bounds.top)

class PlayerHomingBomb(pygame.sprite.Sprite):
    """Homing bomb fired by player that seeks the nearest enemy or the boss and explodes on contact."""
    def __init__(self, x, y, game_ref, speed=420, lifetime=8.0, radius=10, damage=107):
        super().__init__()
        self.game_ref = game_ref
        self.speed = float(speed)
        self.lifetime = float(lifetime)
        self.damage = int(damage)
        self.image = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 230, 120), (radius, radius), radius)
        pygame.draw.circle(self.image, (60, 60, 20), (radius, radius), max(1, radius-4))
        pygame.draw.circle(self.image, (255, 255, 200), (radius, radius), max(1, radius-7))
        self.rect = self.image.get_rect(center=(x, y))
        self._vx = 0.0
        self._vy = -80.0

    def _nearest_target(self):
        enemies = [e for e in self.game_ref.enemy_group.sprites() if e.alive()]
        target = None
        if enemies:
            cx, cy = self.rect.centerx, self.rect.centery
            target = min(enemies, key=lambda e: (e.rect.centerx-cx)**2 + (e.rect.centery-cy)**2)
        elif self.game_ref.boss_alive and self.game_ref.boss_group.sprite:
            target = self.game_ref.boss_group.sprite
        return target

    def update(self, dt, bounds):
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.kill()
            return
        tgt = self._nearest_target()
        if tgt is not None:
            tx, ty = tgt.rect.centerx, tgt.rect.centery
            dx, dy = (tx - self.rect.centerx), (ty - self.rect.centery)
            dist = max(1.0, (dx*dx + dy*dy) ** 0.5)
            ax = (dx / dist) * self.speed
            ay = (dy / dist) * self.speed
            self._vx += ax * dt * 0.9
            self._vy += ay * dt * 0.9
            sp = (self._vx*self._vx + self._vy*self._vy) ** 0.5
            if sp > self.speed:
                self._vx *= (self.speed / sp)
                self._vy *= (self.speed / sp)
        self.rect.x += int(self._vx * dt)
        self.rect.y += int(self._vy * dt)
        if not bounds.inflate(80, 80).colliderect(self.rect):
            self.kill()

class PowerUp(pygame.sprite.Sprite):
    def __init__(self, kind, image, x, y):
        super().__init__()
        self.kind = kind  # 'spread','rapid','shield','bomb','magnet'
        self.image = image.copy() if image else self.make_placeholder(kind)
        self.rect = self.image.get_rect(center=(x,y))
        self.vy = 120
        self.vx = random.uniform(-20, 20)

    def make_placeholder(self, kind):
        surf = pygame.Surface((POWERUP_MAX_SIZE, POWERUP_MAX_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surf, (240,240,240), (POWERUP_MAX_SIZE//2, POWERUP_MAX_SIZE//2), POWERUP_MAX_SIZE//2, 2)
        fnt = pygame.font.SysFont("arial", 14, bold=True)
        label = fnt.render(kind[0].upper(), True, (255,255,255))
        surf.blit(label, label.get_rect(center=(POWERUP_MAX_SIZE//2, POWERUP_MAX_SIZE//2)))
        return surf

    def update(self, dt, bounds, player=None, magnet_active=False):
        # Magnet attraction
        if magnet_active and player is not None:
            px, py = player.rect.centerx, player.rect.centery
            dx = px - self.rect.centerx
            dy = py - self.rect.centery
            dist = max(1.0, math.hypot(dx, dy))
            if dist < 360:
                pull = 4200.0 / dist  # stronger when closer
                self.vx += (dx / dist) * pull * dt
                self.vy += (dy / dist) * pull * dt
                # clamp
                self.vx = clamp(self.vx, -260, 260)
                self.vy = clamp(self.vy, 60, 420)

        self.rect.x += int(self.vx * dt)
        self.rect.y += int(self.vy * dt)
        if self.rect.left < bounds.left or self.rect.right > bounds.right:
            self.vx *= -0.7
        if self.rect.top > bounds.bottom + 50:
            self.kill()

class Player(pygame.sprite.Sprite):
    def __init__(self, image):
        super().__init__()
        self.base_image = image
        self.image = image.copy()
        self.rect = self.image.get_rect()
        self.speed = 420
        self.base_cooldown = 0.2
        self.cooldown = self.base_cooldown
        self.cool_timer = 0.0
        self.lives = 3
        self.hidden = False
        self.hide_timer = 0.0
        # power-ups active times
        self.power_timers = {"spread":0.0,"rapid":0.0,"shield":0.0,"magnet":0.0}

    def reset(self, pos):
        self.rect.center = pos
        self.lives = 3
        self.hidden = False
        self.hide_timer = 0.0
        for k in self.power_timers:
            self.power_timers[k] = 0.0

    def update(self, dt, keys, bounds):
        if self.hidden:
            self.hide_timer -= dt
            if self.hide_timer <= 0:
                self.hidden = False
        dx = dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1
        if dx or dy:
            n = math.hypot(dx,dy)
            dx, dy = dx/n, dy/n
            self.rect.x += int(dx * self.speed * dt)
            self.rect.y += int(dy * self.speed * dt)
            self.rect.clamp_ip(bounds)

        # timers
        for k in list(self.power_timers.keys()):
            if self.power_timers[k] > 0:
                self.power_timers[k] = max(0.0, self.power_timers[k] - dt)

        # dynamic cooldown with rapid
        target_cd = self.base_cooldown * (0.6 if self.power_timers["rapid"] > 0 else 1.0)
        # Smooth towards target for feel
        self.cooldown += (target_cd - self.cooldown) * min(1.0, dt*10)

        if self.cool_timer > 0:
            self.cool_timer -= dt

    def can_shoot(self):
        return not self.hidden and self.cool_timer <= 0

    def shoot(self):
        self.cool_timer = self.cooldown

    
    def hit(self):
        # If a shield (bubble) is active, ignore damage
        if self.power_timers.get("shield", 0.0) > 0.0:
            return False
        # Take damage once, then grant a 3s bubble shield (visible)
        self.lives -= 1
        self.power_timers["shield"] = 3.0  # bubble duration
        # Do NOT hide the player anymore; remain visible with bubble FX
        self.hidden = False
        self.hide_timer = 0.0
        return True

    def has(self, name):
        return self.power_timers.get(name, 0.0) > 0.0

class Enemy(pygame.sprite.Sprite):
    def __init__(self, image, x, y, speed, hp=1, score=10):
        super().__init__()
        self.base_image = image
        self.image = image.copy()
        self.rect = self.image.get_rect(center=(x,y))
        self.speed = speed
        self.hp = hp
        self.max_hp = hp
        self.score_value = score
        self.side_speed = random.uniform(-80, 80)
        self.flash_timer = 0.0
        self.flash_duration = 0.10

    def update(self, dt, bounds):
        self.rect.y += int(self.speed * dt)
        self.rect.x += int(self.side_speed * dt)
        if self.rect.left <= bounds.left:
            self.rect.left = bounds.left
            self.side_speed = abs(self.side_speed)
        elif self.rect.right >= bounds.right:
            self.rect.right = bounds.right
            self.side_speed = -abs(self.side_speed)
        if self.rect.top > bounds.bottom:
            self.kill()
        if self.flash_timer > 0:
            self.flash_timer -= dt
            ratio = max(0.0, min(1.0, self.flash_timer / self.flash_duration))
            glow = self.base_image.copy()
            glow.fill((255,255,255,int(140*ratio)), special_flags=pygame.BLEND_RGBA_MULT)
            self.image = self.base_image.copy()
            self.image.blit(glow, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
        else:
            self.image = self.base_image

    def hit(self, dmg):
        self.hp -= dmg
        self.flash_timer = self.flash_duration
        return self.hp <= 0

class Boss(pygame.sprite.Sprite):
    def __init__(self, image, x, y, hp, speed=140, fire_cooldown=1.0):
        super().__init__()
        self.base_image = image
        self.image = image.copy()
        self.rect = self.image.get_rect(center=(x,y))
        self.hp = hp
        self.max_hp = hp
        self.dir = 1
        self.speed = speed
        self.fire_cooldown = fire_cooldown
        self.cool = 0.0
        self.flash_timer = 0.0
        self.flash_duration = 0.12

    def update(self, dt, bounds):
        self.rect.x += int(self.dir * self.speed * dt)
        if self.rect.left <= bounds.left+20:
            self.dir = 1
        elif self.rect.right >= bounds.right-20:
            self.dir = -1
        if self.cool > 0:
            self.cool -= dt
        if self.flash_timer > 0:
            self.flash_timer -= dt
            ratio = max(0.0, min(1.0, self.flash_timer / self.flash_duration))
            glow = self.base_image.copy()
            glow.fill((255,255,255,int(110*ratio)), special_flags=pygame.BLEND_RGBA_MULT)
            self.image = self.base_image.copy()
            self.image.blit(glow, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
        else:
            self.image = self.base_image

    def ready(self):
        return self.cool <= 0

    def fired(self):
        self.cool = self.fire_cooldown

    def hit(self, dmg):
        self.hp -= dmg
        self.flash_timer = self.flash_duration
        return self.hp <= 0

# ------------------------------ FX ----------------------------------------

class BubbleShield(pygame.sprite.Sprite):
    """Visual bubble that follows the player while shield is active."""
    def __init__(self, player, duration=3.0, color=(130,200,255)):
        super().__init__()
        self.player = player
        self.timer = float(duration)
        self.color = color
        self.image = pygame.Surface((max(1, player.rect.width+24), max(1, player.rect.height+24)), pygame.SRCALPHA)
        self._rebuild()
        self.rect = self.image.get_rect(center=player.rect.center)

    def _rebuild(self):
        w, h = self.image.get_size()
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # soft outer ring
        pygame.draw.ellipse(surf, (*self.color, 70), (0,0,w,h), width=8)
        pygame.draw.ellipse(surf, (*self.color, 40), (6,6,w-12,h-12), width=4)
        self.image = surf

    def update(self, dt, bounds):
        self.timer -= dt
        # Follow the player
        if self.player and self.player.alive():
            self.rect.center = self.player.rect.center
        if self.timer <= 0 or self.player.power_timers.get("shield", 0.0) <= 0.0:
            self.kill()
class DamagePopup(pygame.sprite.Sprite):
    def __init__(self, text, x, y, font, color=(255,240,200)):
        super().__init__()
        self.font = font
        self.base_image = self.font.render(text, True, color)
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.lifetime = 0.6
        self.vy = -80
        self.alpha = 255

    def update(self, dt, bounds):
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.kill()
            return
        self.rect.y += int(self.vy * dt)
        self.alpha = int(255 * (self.lifetime / 0.6))
        self.image = self.base_image.copy()
        self.image.set_alpha(self.alpha)

# ------------------------------ Game Core ----------------------------------
class Game:
    def get_fire_profile(self, n):
        """Return dict with pattern and laser availability for the player at level n."""
        profile = {"pattern": "single", "laser": False}
        if n >= 10:
            profile["pattern"] = "triple"
        elif n >= 7:
            profile["pattern"] = "double_lr"
        elif n >= 5:
            profile["pattern"] = "double_forward"
        else:
            profile["pattern"] = "single"
        if n >= 15:
            profile["laser"] = True
        return profile

    def __init__(self):
        pygame.init()

        flags = pygame.RESIZABLE | pygame.SCALED
        display_info = pygame.display.get_desktop_sizes()
        if display_info:
            desk_w, desk_h = display_info[0]
        else:
            desk_w, desk_h = (BASE_W, BASE_H)

        self.max_size = (desk_w, desk_h)
        self.normal_size = (min(1280, int(desk_w*0.8)), min(720, int(desk_h*0.8)))
        self.is_maximized = True

        try:
            self.screen = pygame.display.set_mode(self.max_size, flags)
        except pygame.error:
            try:
                self.screen = pygame.display.set_mode(self.max_size, pygame.RESIZABLE)
            except pygame.error:
                self.screen = pygame.display.set_mode(self.max_size)
        pygame.display.set_caption(CAPTION)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 18)
        self.bigfont = pygame.font.SysFont("arial", 48, bold=True)
        self.medfont = pygame.font.SysFont("arial", 28, bold=True)

        self.bounds = self.screen.get_rect()

        # Load overlays (background images)
        self.overlays = load_images_from(BG_DIR)
        if not self.overlays:
            self.overlays = [self.make_gradient_surface(1920,1080)]

        # Assets via explicit filenames
        (self.player_img,
         self.enemy_imgs_all,
         self.boss_imgs_all,
         self.powerup_imgs) = load_game_assets()

        self.enemy_imgs_active = [self.enemy_imgs_all[0]] if self.enemy_imgs_all else []
        self.boss_img = self.boss_imgs_all[0] if self.boss_imgs_all else procedural_boss()

        # Groups
        self.player_hazards = pygame.sprite.Group()  # persistent player hazards (laser)
        self.all_sprites = pygame.sprite.LayeredUpdates()
        self.player_group = pygame.sprite.GroupSingle()
        self.enemy_group = pygame.sprite.Group()
        self.boss_group = pygame.sprite.GroupSingle()
        self.player_bullets = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()
        self.powerups = pygame.sprite.Group()
        self.fx_group = pygame.sprite.Group()
        self.hazards = pygame.sprite.Group()

        # Player
        self.player = Player(self.player_img)
        self.player.reset((self.bounds.centerx, self.bounds.bottom - int(self.bounds.h*0.15)))
        self.player_group.add(self.player)
        self.all_sprites.add(self.player, layer=5)

        # UI Buttons
        self.buttons = []
        bx = 10
        bw = 120
        gap = 10
        def add_btn(label, cb):
            nonlocal bx
            b = Button(label, (bx, 6, bw, TOPBAR_H-12), cb)
            self.buttons.append(b)
            bx += bw + gap
        add_btn("Start", self.ui_start)
        add_btn("Pause", self.ui_pause)
        add_btn("Exit", self.ui_exit)
        add_btn("Info", self.ui_info)
        add_btn("Highscores", self.ui_highscores)

        # Game state
        # Pause menu
        self.pause_buttons = []
        self._build_pause_menu()

        self.reset_game()

        # Double-click detection on top bar
        self.last_click_time = 0
        self.double_click_ms = 400

        # Camera shake & slow-mo
        self.shake_timer = 0.0
        self.shake_amp = 0.0
        self.slowmo_timer = 0.0
        self.timescale_slow = 0.25

        # Combo
        self.combo_level = 0  # 0..5 (x0 treated as x1 for display? We'll show x0 as off)
        self.combo_timer = 0.0

    def reset_game(self):
        self.highscore_recorded = False
        self.level = 1
        self.score = 0
        self.player.lives = 3
        self.game_over = False
        self.paused = False
        self.show_info = False
        self.show_hiscores = False
        self.overlay_img = random.choice(self.overlays)
        # Defaults for boss shooting (will be set in prepare_level)
        self.boss_bullet_vx = [0]
        self.boss_bullet_speed = 360
        self.boss_fire_cd = 1.2
        self.boss_speed_val = 140
        self.prepare_level(self.level)

        # Boss new weapons
        self.boss_shoot_strategy = 'forward'  # 'forward','left_right','random','triple'
        self.boss_shooters_count = 1
        self.boss_horiz_speed = 340
        # Laser
        self.boss_laser_available = False
        self.boss_laser_cooldown = 7.0
        self._laser_cd_timer = 0.0
        self._laser_active = False
        self._active_laser_sprite = None
        # Bombs
        self.boss_bomb_available = False
        self.boss_bomb_cooldown = 4.0
        self._bomb_cd_timer = 0.0

        # Magnet power-up auto-bomb cadence
        self._magnet_bomb_timer = 0.0

    def prepare_level(self, n):
        self.enemies_to_clear = 10 + (n-1)*5
        self.enemies_killed = 0
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_rate = max(0.35, 0.9 - n*0.05)
        self.enemy_speed = 110 + n*14
        self.enemy_hp = 1 + (n//3)
        self.boss_pending = True
        self.boss_alive = False
        base_boss_hp = 20 + n*12
        self.boss_hp = base_boss_hp

        # Progressive enemies
        active_count = min(n, len(self.enemy_imgs_all))
        if active_count == 0:
            self.enemy_imgs_active = [procedural_enemy()]
        else:
            self.enemy_imgs_active = self.enemy_imgs_all[:active_count]

        # Boss per 5 levels (cosmetic swap)
        if self.boss_imgs_all:
            boss_index = ((n - 1) // 5) % len(self.boss_imgs_all)
            self.boss_img = self.boss_imgs_all[boss_index]
        else:
            self.boss_img = procedural_boss()

        # ---------------- Boss attack patterns (as requested) ----------------
        # Default movement/speeds scale gently
        self.boss_bullet_speed = 360 + max(0, (n-4)) * 6   # forward (down) bullets
        self.boss_horiz_speed = 320 + max(0, (n-5)) * 5    # left/right bullets
        self.boss_fire_cd = max(0.75, 1.4 - 0.03 * n)      # rate of volleys
        self.boss_speed_val = min(190, 100 + 6 * n)        # boss lateral movement
        # HP scale
        self.boss_hp = max(base_boss_hp, 20 + n * 12)

        # Weapon unlocks by level:
        # Lv1-2: one shooter forward
        # Lv3-4: two shooters forward
        # Lv5-6: two shooters left & right
        # Lv7-9: two shooters that randomly choose directions (each shooter independently forward/left/right)
        # Lv10-14: three shooters (forward + left + right)
        # Lv15+: big laserbeam (persists a couple of seconds)
        # Lv20+: homing bombs that chase player ~5s (player is faster)

        self.boss_laser_available = (n >= 15)
        self.boss_bomb_available  = (n >= 20)

        if n <= 2:
            self.boss_shoot_strategy = 'forward'
            self.boss_shooters_count = 1
        elif n <= 4:
            self.boss_shoot_strategy = 'forward'
            self.boss_shooters_count = 2
        elif n <= 6:
            self.boss_shoot_strategy = 'left_right'
            self.boss_shooters_count = 2
        elif n <= 9:
            self.boss_shoot_strategy = 'random'
            self.boss_shooters_count = 2
        else:
            self.boss_shoot_strategy = 'triple'
            self.boss_shooters_count = 3

        # Player weapon & extras
        self.player_fire_profile = self.get_fire_profile(n)
        self.player_bombs = 3 if n >= 25 else 0
        self._hint_timer = 3.5 if n == 25 else 0.0
        self._hint_text = "NEW: Homing Bombs — press B (3 per level). Great vs bosses!"

        # Reset boss weapon timers
        self._laser_cd_timer = 2.5 if self.boss_laser_available else 0.0  # first laser a bit later
        self._laser_active = False
        self._active_laser_sprite = None
        self._bomb_cd_timer = 2.0 if self.boss_bomb_available else 0.0

        # clear groups (except player)
        for g in [self.enemy_group, self.boss_group, self.player_bullets, self.enemy_bullets, self.powerups, self.fx_group, self.hazards, self.player_hazards]:
            for s in g.sprites():
                s.kill()

        # reset combo on level start
        self.combo_level = 0
        self.combo_timer = 0.0

    # ----------------------- Overlay / Procedural -----------------------
    def make_gradient_surface(self, w, h):
        surf = pygame.Surface((w,h)).convert_alpha()
        for y in range(h):
            t = y / max(1,h-1)
            r = int(30 + 100*t)
            g = int(60 + 120*t)
            b = int(90 + 150*t)
            pygame.draw.line(surf, (r,g,b,255), (0,y), (w,y))
        return surf

    # ----------------------- UI callbacks -------------------------------
    def ui_start(self):
        self.reset_game()

    def ui_pause(self):
        if not self.game_over:
            self.paused = not self.paused

    def ui_exit(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    
    def ui_info(self):
        # Open/close the Info overlay. Opening Info always pauses the game.
        if self.show_info:
            # Close Info; keep whatever paused state we already have
            self.show_info = False
        else:
            self.paused = True
            self.show_info = True


    def ui_highscores(self):
        self.show_hiscores = not self.show_hiscores

    def _build_pause_menu(self):
        # Build simple centered pause menu
        self.pause_buttons = [
            Button("Resume", (0, 0, 260, 48), self.ui_pause),
            Button("Highscores", (0, 0, 260, 48), self.ui_highscores),
            Button("Info", (0, 0, 260, 48), self.ui_info),
            Button("Quit", (0, 0, 260, 48), self.ui_exit),
        ]

    # ------------------------- Highscores -------------------------------
    def load_highscores(self):
        if not os.path.exists(HS_FILE):
            return []
        try:
            with open(HS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[:10]
        except Exception:
            pass
        return []

    def save_highscores(self, scores):
        try:
            with open(HS_FILE, 'w', encoding='utf-8') as f:
                json.dump(scores[:10], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def qualifies_highscore(self, score):
        scores = self.load_highscores()
        if len(scores) < 10:
            return True
        mn = min(s['score'] for s in scores) if scores else 0
        return score > mn

    def add_highscore(self, name, score, level):
        scores = self.load_highscores()
        entry = {
            "name": name[:16],
            "score": int(score),
            "level": int(level),
            "date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        scores.append(entry)
        scores.sort(key=lambda s: s['score'], reverse=True)
        scores = scores[:10]
        self.save_highscores(scores)

    # -------------------------- Resize/Max ------------------------------
    def toggle_maximize(self):
        def safe_set_mode(size, flags):
            w = max(640, int(size[0]))
            h = max(360, int(size[1]))
            try:
                return pygame.display.set_mode((w, h), flags)
            except pygame.error:
                try:
                    return pygame.display.set_mode((w, h), pygame.RESIZABLE)
                except pygame.error:
                    return pygame.display.set_mode((w, h))
        flags = pygame.RESIZABLE | pygame.SCALED
        if self.is_maximized:
            self.screen = safe_set_mode(self.normal_size, flags)
            self.is_maximized = False
        else:
            display_info = pygame.display.get_desktop_sizes()
            if display_info:
                self.max_size = display_info[0]
            self.screen = safe_set_mode(self.max_size, flags)
            self.is_maximized = True
        self.bounds = self.screen.get_rect()

    # --------------------------- Enemy Shooting Config -------------------
    def _configure_enemy_shooting(self, enemy):
        """Configure per-enemy shooting behavior **by enemy sprite tier** (enemy#.png).
        - enemy1.png & enemy2.png: never shoot.
        - enemy3.png–enemy6.png: shoot sometimes (low rate; not fast).
        - enemy7.png–enemy9.png: shoot more often (still not fast).
        - enemy10.png and up: always shoot (steady cadence; still not fast).
        """
        # Determine tier index from image list (surfaces are shared, so identity matches).
        tier = 1
        try:
            tier = self.enemy_imgs_all.index(enemy.base_image) + 1
        except Exception:
            tier = 1  # procedural/fallback => treat as tier 1 (no shots)

        if tier < 3:
            enemy.can_shoot = False
            enemy.shoot_prob = 0.0
            enemy.shoot_interval = (2.8, 3.8)
            enemy.enemy_bullet_speed = 260
        elif tier < 7:
            enemy.can_shoot = True
            enemy.shoot_prob = 0.45     # sometimes
            enemy.shoot_interval = (2.6, 3.6)
            enemy.enemy_bullet_speed = 260
        elif tier < 10:
            enemy.can_shoot = True
            enemy.shoot_prob = 0.80     # a lot more
            enemy.shoot_interval = (1.7, 2.4)
            enemy.enemy_bullet_speed = 270
        else:
            enemy.can_shoot = True
            enemy.shoot_prob = 1.00     # always shoot
            enemy.shoot_interval = (1.1, 1.8)
            enemy.enemy_bullet_speed = 280

        lo, hi = enemy.shoot_interval
        enemy.shoot_timer = random.uniform(lo, hi)

    # --------------------------- Spawning -------------------------------
    def spawn_enemy(self):
        img = random.choice(self.enemy_imgs_active) if self.enemy_imgs_active else procedural_enemy()
        w, h = img.get_size()
        x = random.randint(self.bounds.left + w//2 + 10, self.bounds.right - w//2 - 10)
        y = self.bounds.top - h//2 - 20
        e = Enemy(img, x, y, self.enemy_speed, hp=self.enemy_hp, score=10 + self.level*2)
        self.enemy_group.add(e)
        self.all_sprites.add(e, layer=3)
        self._configure_enemy_shooting(e)

    def spawn_boss(self):
        if self.boss_alive:
            return
        x, y = self.bounds.centerx, self.bounds.top + 120  # a bit higher for room
        b = Boss(self.boss_img, x, y, self.boss_hp, speed=self.boss_speed_val, fire_cooldown=self.boss_fire_cd)
        self.boss_group.add(b)
        self.all_sprites.add(b, layer=4)
        self.boss_alive = True
        self.boss_pending = False

    # --------------------------- Effects & Drops -------------------------
    def add_camera_shake(self, amp=6.0, duration=0.25):
        self.shake_amp = max(self.shake_amp, amp)
        self.shake_timer = max(self.shake_timer, duration)

    def camera_offset(self):
        if self.shake_timer <= 0:
            return (0,0)
        t = self.shake_timer
        factor = max(0.0, min(1.0, t / 0.25))
        amp = self.shake_amp * factor
        return (int(random.uniform(-amp, amp)), int(random.uniform(-amp, amp)))

    
    def spawn_player_bubble(self, duration=3.0):
        """Spawn a bubble FX tied to the player for the given duration."""
        b = BubbleShield(self.player, duration=duration)
        self.fx_group.add(b)
        self.all_sprites.add(b, layer=7)
    def spawn_damage_popup(self, x, y, amount):
        dp = DamagePopup(str(int(amount)), x, y, self.font)
        self.fx_group.add(dp)
        self.all_sprites.add(dp, layer=7)

    def spawn_magnet_bomb(self):
        """Spawn a homing bomb from the player while the magnet power-up is active."""
        if not self.player or self.player.hidden:
            return
        x = self.player.rect.centerx
        y = self.player.rect.top - 6
        # Slightly weaker than manual bombs so the power-up isn't too overpowered
        dmg = 60
        bomb = PlayerHomingBomb(x, y, self, speed=430, lifetime=9.0, radius=9, damage=dmg)
        self.player_bullets.add(bomb)
        self.all_sprites.add(bomb, layer=6)
        # Tiny camera kick so you feel the drop
        self.add_camera_shake(amp=3.0, duration=0.15)

    def roll_powerup_drop(self, is_boss=False):
        # Level-scaled chance
        base = 0.06 + 0.02 * (self.level - 1)  # 6% +2% per level
        chance = clamp(base, 0.06, 0.22)
        if is_boss:
            # Boss: guarantee 1 drop, 30% chance for a second
            count = 1 + (1 if random.random() < 0.30 else 0)
        else:
            count = 1 if random.random() < chance else 0
        kinds = list(POWERUP_FILES.keys())
        drops = []
        for _ in range(count):
            drops.append(random.choice(kinds))
        return drops

    def drop_powerups(self, x, y, is_boss=False):
        for kind in self.roll_powerup_drop(is_boss=is_boss):
            img = self.powerup_imgs.get(kind)
            pu = PowerUp(kind, img, x, y)
            self.powerups.add(pu)
            self.all_sprites.add(pu, layer=2)

    # --------------------------- Combo system ---------------------------
    def combo_hit_event(self):
        # Called when player bullet hits an enemy or boss
        if self.combo_timer > 0:
            self.combo_level = min(COMBO_MAX_LEVEL, self.combo_level + 1)
        else:
            self.combo_level = max(1, self.combo_level)  # start streak
        self.combo_timer = COMBO_STEP_TIME

    def combo_step_decay(self, dt):
        if self.combo_level <= 0:
            self.combo_timer = 0.0
            return
        self.combo_timer -= dt
        if self.combo_timer <= 0:
            self.combo_level -= 1
            if self.combo_level > 0:
                self.combo_timer = COMBO_STEP_TIME
            else:
                self.combo_timer = 0.0

    def combo_reset(self):
        self.combo_level = 0
        self.combo_timer = 0.0

    def fire_player_bomb(self):
        if not self.player or self.player.hidden:
            return
        x, y = self.player.rect.centerx, self.player.rect.top - 4
        dmg = 107
        bomb = PlayerHomingBomb(x, y, self, speed=460, lifetime=9.0, radius=10, damage=dmg)
        self.player_bullets.add(bomb)
        self.all_sprites.add(bomb, layer=6)

    # --------------------------- Loop ----------------------------------
    def run(self):
        entering_name = False
        name_buffer = ""

        while True:
            raw_dt = self.clock.tick(60) / 1000.0
            # timers
            if self.shake_timer > 0:
                self.shake_timer -= raw_dt
                if self.shake_timer <= 0:
                    self.shake_timer = 0
                    self.shake_amp = 0
            if self.slowmo_timer > 0:
                self.slowmo_timer -= raw_dt
                if self.slowmo_timer < 0:
                    self.slowmo_timer = 0

            # combo decay (not affected by slow-mo)
            self.combo_step_decay(raw_dt)

            timescale = self.timescale_slow if self.slowmo_timer > 0 else 1.0
            dt = raw_dt * timescale

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

                for b in self.buttons:
                    b.handle_event(e)

                # Pause menu buttons (active only when paused)
                if self.paused:
                    for b in getattr(self, 'pause_buttons', []):
                        b.handle_event(e)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_b and not self.paused and not self.game_over:
                        if self.level >= 25 and getattr(self, 'player_bombs', 0) > 0:
                            self.fire_player_bomb()
                            self.player_bombs -= 1
                    if e.key == pygame.K_ESCAPE and not self.game_over:
                        if self.show_info:
                            # Close Info overlay first; remain paused
                            self.show_info = False
                        else:
                            self.paused = not self.paused
                    if self.paused and e.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)

                    if self.show_hiscores and not entering_name:
                        if e.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_RETURN, pygame.K_SPACE):
                            self.show_hiscores = False

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if e.pos[1] <= TOPBAR_H:
                        now = pygame.time.get_ticks()
                        if (now - self.last_click_time) <= self.double_click_ms:
                            self.toggle_maximize()
                            self.last_click_time = 0
                        else:
                            self.last_click_time = now

                if entering_name and e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN:
                        nm = name_buffer.strip() or "PLAYER"
                        self.add_highscore(nm, self.score, self.level)
                        self.highscore_recorded = True
                        entering_name = False
                        name_buffer = ""
                        self.show_hiscores = True
                    elif e.key == pygame.K_BACKSPACE:
                        name_buffer = name_buffer[:-1]
                    else:
                        ch = e.unicode
                        if ch and (ch.isalnum() or ch in (' ', '_','-','.')) and len(name_buffer) < 16:
                            name_buffer += ch

            keys = pygame.key.get_pressed()

            if not self.paused and not entering_name and not self.game_over:
                self.update_game(dt, keys)

            self.draw_frame(entering_name, name_buffer)

            if self.game_over and not entering_name and not self.highscore_recorded:
                if self.qualifies_highscore(self.score):
                    entering_name = True
                else:
                    self.show_hiscores = True
                    self.highscore_recorded = True

    def _boss_shooters_for_volley(self, boss):
        """Return a list of (vx, vy, offset_x) tuples for current volley."""
        shooters = []
        bx = boss.rect.centerx
        # compute symmetric offsets for aesthetics
        if self.boss_shooters_count == 1:
            offsets = [0]
        elif self.boss_shooters_count == 2:
            offsets = [-24, 24]
        else:
            offsets = [-26, 0, 26]

        def add_forward(off):
            shooters.append((0, self.boss_bullet_speed, off))

        def add_left(off):
            shooters.append((int(-self.boss_horiz_speed*0.7), int(self.boss_bullet_speed), off))

        def add_right(off):
            shooters.append((int(self.boss_horiz_speed*0.7), int(self.boss_bullet_speed), off))

        if self.boss_shoot_strategy == 'forward':
            for off in offsets[:self.boss_shooters_count]:
                add_forward(off)
        elif self.boss_shoot_strategy == 'left_right':
            # left & right always PLUS a forward (down) shot
            add_forward(0)
            if self.boss_shooters_count == 2:
                add_left(offsets[0]); add_right(offsets[1])
            else:
                add_left(offsets[0]); add_right(offsets[2] if len(offsets)>2 else offsets[1])
        elif self.boss_shoot_strategy == 'random':
            # each shooter independently picks a direction
            dirs = [random.choice(('forward','left','right')) for _ in range(self.boss_shooters_count)]
            for off, d in zip(offsets[:self.boss_shooters_count], dirs):
                if d == 'forward':
                    add_forward(off)
                elif d == 'left':
                    add_left(off)
                else:
                    add_right(off)
        else:  # 'triple'
            # forward + left + right (or best effort with fewer shooters)
            if self.boss_shooters_count >= 3:
                add_left(offsets[0]); add_forward(offsets[1]); add_right(offsets[2])
            elif self.boss_shooters_count == 2:
                add_forward(offsets[0]); add_right(offsets[1])
            else:
                add_forward(0)
        return shooters

    def update_game(self, dt, keys):
        if hasattr(self, '_hint_timer') and self._hint_timer > 0:
            self._hint_timer = max(0.0, self._hint_timer - dt)
        self.player.update(dt, keys, self.bounds.inflate(0, -TOPBAR_H))
        
        if keys[pygame.K_SPACE] and self.player.can_shoot():
            self.player.shoot()
            prof = getattr(self, "player_fire_profile", {"pattern":"single","laser":False})
            if prof.get("laser"):
                beam = PlayerLaser(self.player, width=14, duration=0.32, dps=115)
                self.player_hazards.add(beam)
                self.all_sprites.add(beam, layer=6)
            origin_y = self.player.rect.top - 6
            cx = self.player.rect.centerx
            if self.player.has("spread") and prof.get("pattern") != "double_lr":
                for vx in (-220,-110,0,110,220):
                    b = Bullet(cx, origin_y, dy=-700, damage=1, vx=vx)
                    self.player_bullets.add(b); self.all_sprites.add(b, layer=6)
            else:
                pat = prof.get("pattern", "single")
                if pat == "single":
                    b = Bullet(cx, origin_y, dy=-700, damage=1)
                    self.player_bullets.add(b); self.all_sprites.add(b, layer=6)
                elif pat == "double_forward":
                    for dx in (-10, 10):
                        b = Bullet(cx + dx, origin_y, dy=-700, damage=1)
                        self.player_bullets.add(b); self.all_sprites.add(b, layer=6)
                elif pat == "double_lr":
                    b1 = Bullet(cx - 12, self.player.rect.centery, dy=0, vx=-700, damage=1)
                    b2 = Bullet(cx + 12, self.player.rect.centery, dy=0, vx=700, damage=1)
                    self.player_bullets.add(b1); self.all_sprites.add(b1, layer=6)
                    self.player_bullets.add(b2); self.all_sprites.add(b2, layer=6)
                else:  # triple
                    b = Bullet(cx, origin_y, dy=-700, damage=1)
                    self.player_bullets.add(b); self.all_sprites.add(b, layer=6)
                    b1 = Bullet(cx - 12, self.player.rect.centery, dy=0, vx=-700, damage=1)
                    b2 = Bullet(cx + 12, self.player.rect.centery, dy=0, vx=700, damage=1)
                    self.player_bullets.add(b1); self.all_sprites.add(b1, layer=6)
                    self.player_bullets.add(b2); self.all_sprites.add(b2, layer=6)
        # Spawn enemies
        self.enemy_spawn_timer -= dt
        if self.enemy_spawn_timer <= 0 and not self.boss_alive:
            # Max simultaneous enemies: L1 => 5, then +1 per level
            max_on_screen = 5 + max(0, self.level - 1)
            if len(self.enemy_group) < max_on_screen:
                self.spawn_enemy()
            self.enemy_spawn_timer = self.enemy_spawn_rate

        # Update sprites
        self.enemy_group.update(dt, self.bounds)
        self.player_hazards.update(dt, self.bounds)
        self.player_bullets.update(dt, self.bounds)
        self.enemy_bullets.update(dt, self.bounds)
        self.hazards.update(dt, self.bounds)

        # Regular enemy shooting (enemy3.png and up)
        for e in list(self.enemy_group):
            if not getattr(e, 'can_shoot', False):
                continue
            # countdown and possibly fire
            if not hasattr(e, 'shoot_timer'):
                e.shoot_timer = 2.5
            e.shoot_timer -= dt
            if e.shoot_timer <= 0:
                prob = float(getattr(e, 'shoot_prob', 0.0))
                if prob >= 1.0 or (prob > 0.0 and random.random() < prob):
                    bx, by = e.rect.centerx, e.rect.bottom - 2
                    drift = random.uniform(-40, 40)
                    bspeed = int(getattr(e, 'enemy_bullet_speed', 260))
                    b = Bullet(bx, by, dy=bspeed, color=(255,180,120), radius=4, damage=1, vx=int(drift))
                    self.enemy_bullets.add(b)
                    self.all_sprites.add(b, layer=6)
                # reset timer window
                lo, hi = getattr(e, 'shoot_interval', (2.0, 3.0))
                e.shoot_timer = random.uniform(lo, hi)

        # Power-ups update (magnet effect now repurposed: power-ups just drift)
        for pu in list(self.powerups):
            pu.update(dt, self.bounds, player=self.player, magnet_active=False)

        self.fx_group.update(dt, self.bounds)

        # While the 'magnet' power-up is active, automatically drop homing bombs
        if self.player.has("magnet") and not self.player.hidden:
            # Ensure timer exists
            if not hasattr(self, "_magnet_bomb_timer"):
                self._magnet_bomb_timer = 0.0
            self._magnet_bomb_timer -= dt
            if self._magnet_bomb_timer <= 0:
                self.spawn_magnet_bomb()
                # Slightly faster at higher levels, but clamped so it never goes too crazy
                base_interval = 0.95
                interval = base_interval - 0.02 * max(0, self.level - 1)
                self._magnet_bomb_timer = max(0.40, interval)

        # Boss update & fire
        if self.boss_alive:
            boss = self.boss_group.sprite
            boss.update(dt, self.bounds)

            # Laser management
            if self.boss_laser_available:
                if self._laser_active and (self._active_laser_sprite is None or not self._active_laser_sprite.alive()):
                    # laser ended
                    self._laser_active = False
                    self._active_laser_sprite = None
                    # small delay to next volley
                    boss.cool = max(boss.cool, 0.4)

                if not self._laser_active:
                    self._laser_cd_timer -= dt
                    if self._laser_cd_timer <= 0:
                        # fire laser!
                        laser = BossLaser(boss, width=32, duration=2.5)
                        self.hazards.add(laser)
                        self.all_sprites.add(laser, layer=6)
                        self._active_laser_sprite = laser
                        self._laser_active = True
                        # reset cooldown
                        self._laser_cd_timer = self.boss_laser_cooldown + random.uniform(-1.0, 1.0)

            # Bomb management (independent of laser)
            if self.boss_bomb_available:
                self._bomb_cd_timer -= dt
                if self._bomb_cd_timer <= 0:
                    bx0, by0 = boss.rect.centerx, boss.rect.bottom - 8
                    bomb = HomingBomb(bx0, by0, self.player, speed=300 + min(100, max(0, self.level-20)*6), lifetime=5.0, radius=10)
                    self.enemy_bullets.add(bomb)
                    self.all_sprites.add(bomb, layer=6)
                    # next bomb cooldown
                    self._bomb_cd_timer = self.boss_bomb_cooldown + random.uniform(-0.6, 0.6)

            # Only shoot normal bullets if no active laser (for fairness)
            can_fire_bullets = not self._laser_active
            if can_fire_bullets and boss.ready():
                bx, by = boss.rect.centerx, boss.rect.bottom - 10
                shooters = self._boss_shooters_for_volley(boss)
                for vx, vy, offx in shooters:
                    b = Bullet(bx + offx, by, dy=vy, color=(255,140,140), radius=5, damage=1, vx=vx)
                    self.enemy_bullets.add(b)
                    self.all_sprites.add(b, layer=6)
                boss.fired()

        # Player bullets vs enemies
        if len(self.player_hazards) > 0:
            for e in list(self.enemy_group):
                hits_l = pygame.sprite.spritecollide(e, self.player_hazards, dokill=False)
                if hits_l:
                    laser_dmg = sum(getattr(l, 'dps', 0.0) * dt for l in hits_l)
                    if laser_dmg > 0:
                        self.spawn_damage_popup(e.rect.centerx, e.rect.centery-10, laser_dmg)
                        self.combo_hit_event()
                        if e.hit(int(laser_dmg)):
                            mult = max(1, self.combo_level)
                            self.score += int(e.score_value * mult)
                            self.enemies_killed += 1
                            e.kill()
                            self.drop_powerups(e.rect.centerx, e.rect.centery, is_boss=False)
                            self.add_camera_shake(amp=6, duration=0.2)

        for e in list(self.enemy_group):
            hits = pygame.sprite.spritecollide(e, self.player_bullets, dokill=True)
            if hits:
                total = sum(getattr(h,'damage',1) for h in hits)
                self.spawn_damage_popup(e.rect.centerx, e.rect.centery-10, total)
                self.combo_hit_event()
                if e.hit(total):
                    mult = max(1, self.combo_level)
                    self.score += int(e.score_value * mult)
                    self.enemies_killed += 1
                    e.kill()
                    self.drop_powerups(e.rect.centerx, e.rect.centery, is_boss=False)
                    self.add_camera_shake(amp=6, duration=0.2)

        # Player bullets vs boss
        if self.boss_alive and self.boss_group.sprite and len(self.player_hazards) > 0:
            boss = self.boss_group.sprite
            hits_l = pygame.sprite.spritecollide(boss, self.player_hazards, dokill=False)
            if hits_l:
                laser_dmg = sum(getattr(l, 'dps', 0.0) * dt for l in hits_l)
                if laser_dmg > 0:
                    self.spawn_damage_popup(boss.rect.centerx, boss.rect.top, laser_dmg)
                    self.combo_hit_event()
                    if boss.hit(int(laser_dmg)):
                        mult = max(1, self.combo_level)
                        self.score += int((250 + self.level*50) * mult)
                        self.drop_powerups(boss.rect.centerx, boss.rect.centery, is_boss=True)
                        boss.kill()
                        self.boss_alive = False
                        self.add_camera_shake(amp=12, duration=0.35)
                        self.level += 1
                        self.overlay_img = random.choice(self.overlays)
                        self.prepare_level(self.level)

        if self.boss_alive and self.boss_group.sprite:
            boss = self.boss_group.sprite
            hits = pygame.sprite.spritecollide(boss, self.player_bullets, dokill=True)
            if hits:
                total = sum(getattr(h,'damage',1) for h in hits)
                self.spawn_damage_popup(boss.rect.centerx, boss.rect.top, total)
                self.combo_hit_event()
                if boss.hit(total):
                    mult = max(1, self.combo_level)
                    self.score += int((250 + self.level*50) * mult)
                    # Boss death effects
                    self.drop_powerups(boss.rect.centerx, boss.rect.centery, is_boss=True)
                    boss.kill()
                    self.boss_alive = False
                    self.add_camera_shake(amp=12, duration=0.35)
                    self.level += 1
                    self.overlay_img = random.choice(self.overlays)
                    self.prepare_level(self.level)

        # Enemies colliding with player or passing bottom
        for e in list(self.enemy_group):
            if e.rect.colliderect(self.player.rect) and not self.player.hidden:
                e.kill()
                if self.player.hit():  # returns True if took damage
                    self.spawn_player_bubble(duration=3.0)
                    self.slowmo_timer = 0.2
                    self.combo_reset()
                    if self.player.lives <= 0:
                        self.game_over = True
            elif e.rect.top > self.bounds.bottom:
                e.kill()
                # reset combo if a plane escapes
                self.combo_reset()

        # Enemy bullets vs player
        if not self.player.hidden:
            hits = pygame.sprite.spritecollide(self.player, self.enemy_bullets, dokill=True)
            if hits:
                if self.player.hit():
                    self.spawn_player_bubble(duration=3.0)
                    self.slowmo_timer = 0.2
                    self.combo_reset()
                    if self.player.lives <= 0:
                        self.game_over = True

        # Hazards (e.g., laser) vs player — do not kill the hazard
        if not self.player.hidden and len(self.hazards) > 0:
            hits_h = pygame.sprite.spritecollide(self.player, self.hazards, dokill=False)
            if hits_h:
                if self.player.hit():
                    self.spawn_player_bubble(duration=3.0)
                    self.spawn_player_bubble(duration=3.0)
                    self.slowmo_timer = 0.2
                    self.combo_reset()
                    if self.player.lives <= 0:
                        self.game_over = True

        # Power-up pickups
        for pu in list(self.powerups):
            if pu.rect.colliderect(self.player.rect) and not self.player.hidden:
                self.apply_powerup(pu.kind)
                pu.kill()

        # Boss spawn condition
        if self.boss_pending and self.enemies_killed >= self.enemies_to_clear:
            self.spawn_boss()

    def apply_powerup(self, kind):
        # Sound/flash hooks could be here
        if kind in POWERUP_DURATIONS:
            self.player.power_timers[kind] = POWERUP_DURATIONS[kind]
            if kind == "magnet":
                # Start dropping homing bombs right away
                self._magnet_bomb_timer = 0.0
        elif kind == "bomb":
            # Instant: clear enemy bullets and damage all enemies & boss
            cleared = 0
            for b in list(self.enemy_bullets):
                b.kill()
                cleared += 1
            dmg_each = 3
            for e in list(self.enemy_group):
                if e.hit(dmg_each):
                    mult = max(1, self.combo_level)
                    self.score += int(e.score_value * mult)
                    self.enemies_killed += 1
                    self.drop_powerups(e.rect.centerx, e.rect.centery, is_boss=False)
                    e.kill()
            if self.boss_alive and self.boss_group.sprite:
                boss = self.boss_group.sprite
                chunk = max(8, int(boss.max_hp * 0.15))
                if boss.hit(chunk):
                    mult = max(1, self.combo_level)
                    self.score += int((250 + self.level*50) * mult)
                    self.drop_powerups(boss.rect.centerx, boss.rect.centery, is_boss=True)
                    boss.kill()
                    self.boss_alive = False
                    self.level += 1
                    self.overlay_img = random.choice(self.overlays)
                    self.prepare_level(self.level)
            self.add_camera_shake(amp=10, duration=0.35)

    # ----------------------------- Draw ---------------------------------
    def draw_topbar(self, surf):
        bar = pygame.Rect(0,0,self.bounds.w, TOPBAR_H)
        pygame.draw.rect(surf, (20,20,20), bar)
        pygame.draw.line(surf, (80,80,80), (0, TOPBAR_H-1), (self.bounds.w, TOPBAR_H-1))

        for b in self.buttons:
            b.draw(surf, self.font)

        extra = ""
        if getattr(self, "level", 1) >= 25:
            extra = f"   Bombs: {getattr(self, 'player_bombs', 0)} (B)"
        txt = f"Score: {self.score}   Lives: {self.player.lives}   Level: {self.level}" + extra
        info = self.font.render(txt, True, (230,230,230))
        surf.blit(info, (self.bounds.w - info.get_width() - 12, (TOPBAR_H-info.get_height())//2))

    def draw_boss_hpbar(self, surf):
        if self.boss_alive and self.boss_group.sprite:
            boss = self.boss_group.sprite
            frac = max(0.0, min(1.0, boss.hp / max(1, boss.max_hp)))
            bar_w = int(self.bounds.w * 0.45)
            bar_h = 10
            x = 12
            y = TOPBAR_H + 6  # below the buttons
            pygame.draw.rect(surf, (50,50,50), (x, y, bar_w, bar_h), border_radius=5)
            pygame.draw.rect(surf, (220,60,60), (x, y, int(bar_w * frac), bar_h), border_radius=5)
            label = self.font.render("BOSS", True, (230,200,200))
            surf.blit(label, (x+6, y-18))

    def draw_powerup_timers(self, surf):
        """Mini HUD: show active power-up timers under the top bar while playing."""
        # Position to the right of the boss HP bar
        start_x = int(self.bounds.w * 0.45) + 40
        x = start_x
        y = TOPBAR_H + 2
        icon_size = 28

        for kind in ("spread", "rapid", "shield", "magnet"):
            t = self.player.power_timers.get(kind, 0.0)
            if t <= 0:
                continue  # only show active power-ups

            img = self.powerup_imgs.get(kind)
            icon = scale_to_box(img, icon_size) if img else None

            # Draw icon or a simple placeholder box with the first letter
            if icon:
                rect = icon.get_rect(topleft=(x, y))
                surf.blit(icon, rect)
                iw, ih = rect.w, rect.h
            else:
                iw = ih = icon_size
                rect = pygame.Rect(x, y, iw, ih)
                pygame.draw.rect(surf, (80, 80, 80), rect, border_radius=6)
                pygame.draw.rect(surf, (140, 140, 140), rect, 1, border_radius=6)
                letter = kind[0].upper()
                lbl = self.font.render(letter, True, (230, 230, 230))
                lw, lh = lbl.get_size()
                surf.blit(lbl, (x + (iw - lw) // 2, y + (ih - lh) // 2))

            # Progress bar under the icon
            dur = POWERUP_DURATIONS.get(kind, max(0.1, t))
            frac = clamp(t / dur, 0.0, 1.0)
            bar_w = iw
            bar_h = 4
            bx = x
            by = y + ih + 2
            pygame.draw.rect(surf, (40, 40, 40), (bx, by, bar_w, bar_h), border_radius=2)
            pygame.draw.rect(surf, (160, 220, 160), (bx, by, int(bar_w * frac), bar_h), border_radius=2)

            # Label: show power-up name above the bar
            name_map = {
                "spread": "Spread",
                "rapid": "Rapid",
                "shield": "Shield",
                "magnet": "Homing Bombs",
            }
            label_text = name_map.get(kind, kind.title())
            label_surf = self.font.render(label_text, True, (220, 240, 220))
            tw, th = label_surf.get_size()
            # Center the label over the bar if it fits; otherwise left-align
            if tw <= bar_w:
                label_x = bx + (bar_w - tw) // 2
            else:
                label_x = bx
            surf.blit(label_surf, (label_x, by - th - 1))

            x += iw + 12


    def draw_combo_bar(self, surf):
        # Right-aligned small bar with multiplier
        if self.combo_level <= 0:
            return
        bar_w = 150
        bar_h = 10
        x = self.bounds.w - bar_w - 12
        y = TOPBAR_H + 24  # under the boss bar line area
        # background
        pygame.draw.rect(surf, (40,40,60), (x, y, bar_w, bar_h), border_radius=5)
        # filled by current timer fraction
        frac = clamp(self.combo_timer / COMBO_STEP_TIME, 0.0, 1.0)
        pygame.draw.rect(surf, (120,200,255), (x, y, int(bar_w*frac), bar_h), border_radius=5)
        # outline
        pygame.draw.rect(surf, (90,90,120), (x, y, bar_w, bar_h), 1, border_radius=5)
        # label
        label = self.font.render(f"x{self.combo_level}", True, (220,240,255))
        surf.blit(label, (x - label.get_width() - 8, y - 2))

    def draw_overlay_image(self, surf, ofsx, ofsy):
        img = self.overlay_img
        if img:
            sw, sh = self.bounds.size
            scaled = pygame.transform.smoothscale(img, (sw, sh))
            scaled.set_alpha(BG_OVERLAY_ALPHA)
            surf.blit(scaled, (ofsx, ofsy))

    def draw_info_overlay(self, surf):
        # Dim the scene
        shade = pygame.Surface(self.bounds.size, pygame.SRCALPHA)
        shade.fill((0,0,0,150))
        surf.blit(shade, (0,0))

        # Title
        title = self.bigfont.render("INFO", True, (255,255,255))
        surf.blit(title, title.get_rect(center=(self.bounds.centerx, int(self.bounds.h*0.18))))

        x = self.bounds.centerx - 460
        y = int(self.bounds.h*0.24)

        # Controls (compact)
        controls = [
            "Move: Arrows / WASD   •   Shoot: SPACE   •   Pause: ESC",
            "Paused menu: click Resume / Highscores / Info / Quit"
        ]
        for line in controls:
            label = self.medfont.render(line, True, (230,230,230))
            surf.blit(label, (x, y)); y += 30

        # Separator
        y += 10
        pygame.draw.line(surf, (90,90,90), (x, y), (x+900, y), 1)
        y += 18

        # Power-ups panel
        hdr = self.medfont.render("Power-Ups", True, (255,255,255))
        surf.blit(hdr, (x, y)); y += 42

        kinds = ("spread","rapid","shield","magnet")
        row_h = 54
        icon_size = 36
        bar_w = 420
        bar_h = 8

        for k in kinds:
            # Icon
            img = self.powerup_imgs.get(k)
            icon = scale_to_box(img, icon_size) if img else None
            ix = x + 8
            iy = y + (row_h - (icon.get_height() if icon else icon_size))//2

            if icon:
                # If inactive, dim the icon
                t = self.player.power_timers.get(k, 0.0)
                if t <= 0:
                    dim = pygame.Surface(icon.get_size(), pygame.SRCALPHA)
                    dim.fill((0,0,0,120))
                    surf.blit(icon, (ix, iy))
                    surf.blit(dim, (ix, iy))
                else:
                    surf.blit(icon, (ix, iy))
            else:
                # Fallback box
                pygame.draw.rect(surf, (80,80,80), (ix, iy, icon_size, icon_size), border_radius=6)

            # Label and timer
            name_txt = k.title()
            t = self.player.power_timers.get(k, 0.0)
            dur = POWERUP_DURATIONS.get(k, 10.0)
            lbl = self.medfont.render(name_txt, True, (230,230,230))
            surf.blit(lbl, (ix + icon_size + 14, y + 2))

            # Timer text
            if t > 0:
                mm = int(t // 60); ss = int(t % 60)
                ttxt = f"{mm}:{ss:02d} remaining"
                tcol = (180, 240, 180)
            else:
                ttxt = "inactive"
                tcol = (180, 180, 180)
            tlabel = self.font.render(ttxt, True, tcol)
            surf.blit(tlabel, (ix + icon_size + 14, y + 26))

            # Progress bar
            bx = ix + icon_size + 14 + 240
            by = y + 28
            pygame.draw.rect(surf, (50,50,50), (bx, by, bar_w, bar_h), border_radius=3)
            frac = 0.0 if t <= 0 else max(0.0, min(1.0, t / dur))
            pygame.draw.rect(surf, (120, 200, 140), (bx, by, int(bar_w * frac), bar_h), border_radius=3)

            y += row_h

        # Separator
        y += 6
        pygame.draw.line(surf, (90,90,90), (x, y), (x+900, y), 1)
        y += 14

        # Loadout
        hdr2 = self.medfont.render("Loadout", True, (255,255,255))
        surf.blit(hdr2, (x, y)); y += 38

        pattern = getattr(self, "player_fire_profile", {}).get("pattern", "single")
        laser_av = getattr(self, "player_fire_profile", {}).get("laser", False)
        bombs = getattr(self, "player_bombs", 0)

        rows = [
            f"Fire pattern: {pattern}",
            f"Laser: {'available' if laser_av else 'unavailable'}",
            f"Bombs: {bombs}"
        ]
        for r in rows:
            label = self.medfont.render(r, True, (230,230,230))
            surf.blit(label, (x, y)); y += 28

        # Footer tip
        y += 8
        foot = "Collect power-ups to refresh timers. Icons + mini bars also appear under the top bar."
        flabel = self.font.render(foot, True, (210,210,210))
        surf.blit(flabel, (x, y))

    def draw_highscores_overlay(self, surf):
        shade = pygame.Surface(self.bounds.size, pygame.SRCALPHA)
        shade.fill((0,0,0,140))
        surf.blit(shade, (0,0))

        title = self.bigfont.render("HIGHSCORES", True, (255,255,255))
        surf.blit(title, title.get_rect(center=(self.bounds.centerx, self.bounds.h//6)))

        scores = self.load_highscores()
        y = self.bounds.h//6 + 80
        if not scores:
            label = self.medfont.render("No highscores yet. Go play! :)", True, (230,230,230))
            surf.blit(label, label.get_rect(center=(self.bounds.centerx, y)))
            return

        x1 = self.bounds.centerx - 280
        for i, s in enumerate(scores[:10], 1):
            row = f"{i:2d}. {s['name']:<16}  Score: {s['score']:<6}  Lv {s['level']:<2}  ({s['date']})"
            label = self.medfont.render(row, True, (230,230,230))
            surf.blit(label, (x1, y))
            y += 36

    
    def draw_pause_overlay(self, surf):
        shade = pygame.Surface(self.bounds.size, pygame.SRCALPHA)
        shade.fill((0,0,0,160))
        surf.blit(shade, (0,0))

        # Title
        t1 = self.bigfont.render("PAUSED", True, (255,255,255))
        surf.blit(t1, t1.get_rect(center=(self.bounds.centerx, self.bounds.h//2 - 120)))

        # Layout buttons centered
        btn_w, btn_h, gap = 260, 48, 12
        buttons = getattr(self, 'pause_buttons', [])
        total_h = len(buttons)*btn_h + (len(buttons)-1)*gap if buttons else 0
        start_y = self.bounds.h//2 - total_h//2
        x = self.bounds.centerx - btn_w//2

        for i, b in enumerate(buttons):
            b.rect = pygame.Rect(x, start_y + i*(btn_h+gap), btn_w, btn_h)
            b.draw(surf, self.medfont)

        # Hint text
        hint = self.font.render("ESC: resume • Q: quit • or click a menu item", True, (230,230,230))
        surf.blit(hint, hint.get_rect(center=(self.bounds.centerx, start_y + total_h + 36)))

    def draw_name_entry(self, surf, buf):
        shade = pygame.Surface(self.bounds.size, pygame.SRCALPHA)
        shade.fill((0,0,0,160))
        surf.blit(shade, (0,0))
        t1 = self.bigfont.render("NEW HIGHSCORE!", True, (255,255,255))
        surf.blit(t1, t1.get_rect(center=(self.bounds.centerx, self.bounds.h//2 - 80)))
        t2 = self.medfont.render("Enter your name (max 16) and press ENTER:", True, (230,230,230))
        surf.blit(t2, t2.get_rect(center=(self.bounds.centerx, self.bounds.h//2 - 20)))
        box = pygame.Rect(0,0, 420, 46)
        box.center = (self.bounds.centerx, self.bounds.h//2 + 30)
        pygame.draw.rect(surf, (240,240,240), box, border_radius=8)
        pygame.draw.rect(surf, (50,50,50), box.inflate(-6,-6), border_radius=6)
        name = self.medfont.render(buf or "", True, (255,255,255))
        surf.blit(name, (box.x+12, box.y+8))

    def draw_frame(self, entering_name, name_buffer):
        self.bounds = self.screen.get_rect()
        self.screen.fill((8,10,16))

        ofsx, ofsy = self.camera_offset()

        # Background (behind gameplay)
        self.draw_overlay_image(self.screen, ofsx, ofsy)

# Gameplay (with shake)
        for spr in self.all_sprites:
            if spr == self.player and self.player.hidden:
                continue
            self.screen.blit(spr.image, spr.rect.move(ofsx, ofsy))
        # UI (not shaken)
        self.draw_topbar(self.screen)
        self.draw_boss_hpbar(self.screen)
        self.draw_powerup_timers(self.screen)
        self.draw_combo_bar(self.screen)

        if self.paused and not self.game_over:
            self.draw_pause_overlay(self.screen)
        if self.show_info:
            self.draw_info_overlay(self.screen)
        if self.show_hiscores:
            self.draw_highscores_overlay(self.screen)
        # Temporary hint (e.g., B key bombs at L25)
        if hasattr(self, '_hint_timer') and self._hint_timer > 0 and hasattr(self, '_hint_text'):
            msg = self.medfont.render(self._hint_text, True, (255,255,255))
            pad = 14
            box = msg.get_rect(center=(self.bounds.centerx, self.bounds.h - 80))
            bg = pygame.Rect(box.x-pad, box.y-6, box.w+pad*2, box.h+12)
            shade = pygame.Surface((bg.w, bg.h), pygame.SRCALPHA)
            shade.fill((0,0,0,160))
            self.screen.blit(shade, bg)
            self.screen.blit(msg, box)

        if entering_name:
            self.draw_name_entry(self.screen, name_buffer)

        pygame.display.flip()

# ------------------------------ Main ---------------------------------------

def main():
    Game().run()

if __name__ == "__main__":
    main()
