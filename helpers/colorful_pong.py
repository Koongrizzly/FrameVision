#!/usr/bin/env python3
# Colorful Pong — Last-Hit Power-Ups + friendly tokens (v1.2.1)
# Buttons always visible; help/high-scores overlays pause/resume with ESC;
# high scores saved to ./presets/setsave/; F11 & double-click top bar = maximize;
# level label lowered below pause button.

import warnings
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)
import pygame, math, random, json, os, sys

WIDTH, HEIGHT = 960, 540
FPS = 120
TOPBAR_H = 36
PADDLE_W, PADDLE_H = 16, 110
BALL_SIZE = 14
START_SPEED = 290.0
LEVEL_UP_EVERY = 5
MAX_LEVEL = 20
AI_BASE_SPEED = 250.0
AI_REACTION = 0.12

DATA_DIR = os.path.join(os.path.dirname(__file__), "presets", "setsave")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "pong_highscores.json")

POWERUP_SIZE = 28
POWERUP_PICKUP_PAD = 20
POWERUP_ATTRACT_RADIUS = 120
POWERUP_ATTRACT_SPEED = 120.0
POWERUP_COOLDOWN_MIN = 6.0
POWERUP_COOLDOWN_MAX = 10.0
POWERUP_TYPES = ("GROW","SHRINK_OPP","SLOW_BALL","MULTI","MAGNET","CURVE","SHIELD")
POWERUP_EXPLAINS = {
    "GROW":"Your paddle grows for a while.",
    "SHRINK_OPP":"Opponent's paddle shrinks for a while.",
    "SLOW_BALL":"Balls move slower for a while.",
    "MULTI":"Adds one more ball!",
    "MAGNET":"Ball is gently pulled towards your paddle.",
    "CURVE":"Curved hits add spin!",
    "SHIELD":"A wall protects your goal for a while.",
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

    state="MENU"; bg_phase=0.0
    player=pygame.Rect(40, HEIGHT//2-PADDLE_H//2, PADDLE_W, PADDLE_H)
    ai=pygame.Rect(WIDTH-40-PADDLE_W, HEIGHT//2-PADDLE_H//2, PADDLE_W, PADDLE_H)

    balls=[]
    def make_ball(direction=1, speed_scale=1.0, angle_offset=0.0):
        speed=START_SPEED*(1.0+0.08*(level-1))*speed_scale; ang=random.uniform(-0.35,0.35)+angle_offset
        vx=speed*direction*math.cos(ang); vy=speed*math.sin(ang); rect=pygame.Rect(WIDTH//2-BALL_SIZE//2, HEIGHT//2-BALL_SIZE//2, BALL_SIZE, BALL_SIZE)
        return {"rect":rect,"vx":vx,"vy":vy,"spin":0.0,"prev":rect.center,"last":None}

    player_trail=Trail(18); ai_trail=Trail(18); particles=[]; textfx=TextFX(mid); bigfx=BigPopupFX(big, small)

    player_score=0; ai_score=0; level=1; ai_speed=AI_BASE_SPEED; shake=0.0
    powerup=None; next_powerup_time=random.uniform(POWERUP_COOLDOWN_MIN, POWERUP_COOLDOWN_MAX); elapsed=0.0
    paused=False; show_help=False; show_scores=False

    effects={"player_grow":0.0,"ai_grow":0.0,"player_shrink":0.0,"ai_shrink":0.0,"slow_ball":0.0,"player_magnet":0.0,"ai_magnet":0.0,"player_curve":0.0,"ai_curve":0.0,"player_shield":0.0,"ai_shield":0.0}
    original_sizes={"player_h":PADDLE_H,"ai_h":PADDLE_H}
    def apply_sizes():
        ph=original_sizes["player_h"]; ah=original_sizes["ai_h"]
        if effects["player_grow"]>0.0: ph=int(original_sizes["player_h"]*1.4)
        if effects["ai_grow"]>0.0: ah=int(original_sizes["ai_h"]*1.4)
        if effects["player_shrink"]>0.0: ph=int(original_sizes["player_h"]*0.65)
        if effects["ai_shrink"]>0.0: ah=int(original_sizes["ai_h"]*0.65)
        player.h=ph; ai.h=ah; player.y=max(10,min(HEIGHT-player.h-10,player.y)); ai.y=max(10,min(HEIGHT-ai.h-10,ai.y))
    def level_up():
        nonlocal level, ai_speed
        if level<MAX_LEVEL:
            level+=1; ai_speed=AI_BASE_SPEED*(1.0+0.06*(level-1))
            textfx.add(f"Level {level}!", (WIDTH//2, TOPBAR_H+60), rainbow(level*0.2), 1.1, rise=-60)

    def emit_spark(x,y,color,count=22):
        for _ in range(count): particles.append(Particle(x,y,color))

    def start_game():
        nonlocal player_score, ai_score, level, ai_speed, shake, state, balls, powerup, next_powerup_time, elapsed, effects, paused, show_help, show_scores
        player_score=0; ai_score=0; level=1; ai_speed=AI_BASE_SPEED; shake=0.0
        balls=[make_ball(direction=random.choice([-1,1]))]; powerup=None; next_powerup_time=random.uniform(POWERUP_COOLDOWN_MIN, POWERUP_COOLDOWN_MAX); elapsed=0.0
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
        COLORS={"GROW":(100,255,120),"SHRINK_OPP":(255,120,120),"SLOW_BALL":(120,180,255),"MULTI":(255,210,120),"MAGNET":(180,255,180),"CURVE":(255,180,255),"SHIELD":(180,255,255)}
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
            elif self.kind=="MAGNET": pygame.draw.circle(surf,(25,25,25),(cx,cy),8,2); pygame.draw.rect(surf,(25,25,25),(cx-2,cy-12,4,8))
            elif self.kind=="CURVE": pygame.draw.arc(surf,(25,25,25),(cx-10,cy-10,20,20),0.3,2.6,3)
            elif self.kind=="SHIELD": pygame.draw.rect(surf,(25,25,25),(cx-8,cy-10,16,20),2)

    def spawn_powerup():
        nonlocal powerup
        kind=random.choice(POWERUP_TYPES); margin=80
        x=random.randint(WIDTH//3, WIDTH*2//3-POWERUP_SIZE); y=random.randint(margin, HEIGHT-margin-POWERUP_SIZE)
        powerup=PowerUp(kind, x, y)

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
                    direction=1 if balls[0]["vx"]>=0 else -1; angle_offset=random.uniform(-0.35,0.35); balls.append(make_ball(direction=direction, angle_offset=angle_offset))
            elif kind=="MAGNET": effects["player_magnet"]=6.0
            elif kind=="CURVE": effects["player_curve"]=8.0
            elif kind=="SHIELD": effects["player_shield"]=8.0
            textfx.add(labelify(kind), (WIDTH*0.33, TOPBAR_H+56), (255,255,255), 0.9, rise=-50); announce_pickup(kind,"player")
        else:
            if kind=="GROW": effects["ai_grow"]=6.0
            elif kind=="SHRINK_OPP": effects["player_shrink"]=6.0
            elif kind=="SLOW_BALL": effects["slow_ball"]=5.0
            elif kind=="MULTI":
                if balls:
                    direction=-1 if balls[0]["vx"]>=0 else 1; angle_offset=random.uniform(-0.35,0.35); balls.append(make_ball(direction=direction, angle_offset=angle_offset))
            elif kind=="MAGNET": effects["ai_magnet"]=6.0
            elif kind=="CURVE": effects["ai_curve"]=8.0
            elif kind=="SHIELD": effects["ai_shield"]=8.0
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
                                elif name=="EXIT":
                                    if state=="PLAY": finish_game()
                                    else: running=False

        # Background
        canvas.fill((0,0,0,0)); grad=pygame.Surface((WIDTH,HEIGHT))
        for y in range(0,HEIGHT,6): pygame.draw.rect(grad, rainbow((bg_phase*0.2 + y/HEIGHT)*1.5), (0,y,WIDTH,6))
        canvas.blit(grad,(0,0))
        for y in range(0,HEIGHT,28): pygame.draw.rect(canvas,(255,255,255,120),(WIDTH//2-3,y,6,18))

        keys=pygame.key.get_pressed()

        # Always show top bar (ALL states)
        button_rects={}; bar=pygame.Surface((WIDTH,TOPBAR_H), pygame.SRCALPHA); bar.fill((0,0,0,110)); canvas.blit(bar,(0,0))
        labels=[("PAUSE","Resume" if (state=="PLAY" and paused and not (show_help or show_scores)) else "Pause"),
                ("HELP","Help"),("SCORES","High Scores"),("EXIT","Exit")]
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
                target_y=min(balls,key=lambda b:abs(b["rect"].centerx-ai.centerx))["rect"].centery if balls else HEIGHT//2
                dy=target_y-ai.centery; ai.y+=int(max(-1,min(1,dy*AI_REACTION))*ai_speed*dt); ai.y=max(10,min(HEIGHT-ai.h-10,ai.y))
                if powerup is None and elapsed>=next_powerup_time: spawn_powerup()
                if powerup: powerup.update(dt, balls)
                for k in list(effects.keys()):
                    if effects[k]>0.0: effects[k]=max(0.0, effects[k]-dt)
                apply_sizes()
            left_shield=pygame.Rect(6,0,8,HEIGHT) if effects["player_shield"]>0.0 else None
            right_shield=pygame.Rect(WIDTH-14,0,8,HEIGHT) if effects["ai_shield"]>0.0 else None
            slow_factor=0.7 if effects["slow_ball"]>0.0 else 1.0; magnet_accel=220.0
            for b in balls:
                bx,by=b["rect"].center
                if state=="PLAY" and do_update:
                    b["prev"]=(bx,by)
                    if effects["player_magnet"]>0.0:
                        dx=player.centerx-bx; dyb=player.centery-by; d=max(1.0,math.hypot(dx,dyb))
                        b["vx"]+=(dx/d)*magnet_accel*dt; b["vy"]+=(dyb/d)*magnet_accel*dt
                    if effects["ai_magnet"]>0.0:
                        dx=ai.centerx-bx; dyb=ai.centery-by; d=max(1.0,math.hypot(dx,dyb))
                        b["vx"]+=(dx/d)*magnet_accel*dt; b["vy"]+=(dyb/d)*magnet_accel*dt
                    if abs(b.get("spin",0.0))>0.1: b["vy"]+=b["spin"]*dt; b["spin"]*=0.985
                    b["rect"].x+=int(b["vx"]*slow_factor*dt); b["rect"].y+=int(b["vy"]*slow_factor*dt)
                    if left_shield and b["vx"]<0 and b["rect"].colliderect(left_shield): b["rect"].left=left_shield.right; b["vx"]*=-1
                    if right_shield and b["vx"]>0 and b["rect"].colliderect(right_shield): b["rect"].right=right_shield.left; b["vx"]*=-1
                    if b["rect"].top<=0: b["rect"].top=0; b["vy"]*=-1
                    if b["rect"].bottom>=HEIGHT: b["rect"].bottom=HEIGHT; b["vy"]*=-1
                    if b["vx"]<0 and b["rect"].colliderect(player):
                        offset=(b["rect"].centery-player.centery)/(player.h/2)
                        b["vx"]=abs(b["vx"])*1.03+12*level; b["vy"]=(b["vy"]*0.6)+offset*(280+14*level); b["last"]="player"
                        if effects["player_curve"]>0.0: b["spin"]=280.0*offset; textfx.add("Curve!", (WIDTH*0.35, TOPBAR_H+74), (255,220,255), 0.6, rise=-40)
                    elif b["vx"]>0 and b["rect"].colliderect(ai):
                        offset=(b["rect"].centery-ai.centery)/(ai.h/2)
                        b["vx"]=-abs(b["vx"])*1.03-12*level; b["vy"]=(b["vy"]*0.6)+offset*(280+14*level); b["last"]="ai"
                        if effects["ai_curve"]>0.0: b["spin"]=280.0*offset
            if state=="PLAY" and do_update and powerup:
                pick_rect=powerup.rect.inflate(POWERUP_PICKUP_PAD, POWERUP_PICKUP_PAD); collected=False; owner=None
                for b in balls:
                    if b["rect"].colliderect(pick_rect) or segment_intersects_rect(b["prev"], b["rect"].center, pick_rect):
                        collected=True; owner=b["last"] or ("player" if b["vx"]<0 else "ai"); break
                    bx,by=b["rect"].center; cx,cy=powerup.rect.center
                    if (bx-cx)**2+(by-cy)**2 <= POWERUP_SIZE**2: collected=True; owner=b["last"] or ("player" if b["vx"]<0 else "ai"); break
                if not collected and player.colliderect(pick_rect): collected=True; owner="player"
                if not collected and ai.colliderect(pick_rect): collected=True; owner="ai"
                if collected:
                    kind=powerup.kind; award(kind, owner)
                    for _ in range(36): particles.append(Particle(powerup.rect.centerx, powerup.rect.centery, powerup.color))
                    powerup=None; elapsed=0.0; next_powerup_time=random.uniform(POWERUP_COOLDOWN_MIN, POWERUP_COOLDOWN_MAX)
            if state=="PLAY" and do_update:
                scored_left=0; scored_right=0; new_balls=[]
                for b in balls:
                    if b["rect"].right<0: scored_right+=1; continue
                    elif b["rect"].left>WIDTH: scored_left+=1; continue
                    new_balls.append(b)
                balls=new_balls
                if scored_right>0: ai_score+=scored_right; 
                if scored_left>0:
                    player_score+=scored_left
                    if player_score % LEVEL_UP_EVERY==0: level_up()
                if not balls: balls.append(make_ball(direction=-1 if scored_left>0 else 1))
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
